import torch

from clarity.predictor.torch_stoi import NegSTOILoss


class SISNRLoss(torch.nn.Module):
    # def __init__(self):  # removed as does nothing
    #    super(SISNRLoss, self).__init__()

    def cal_sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                f"Dimension mismatch when calculate si-snr, {x.shape} vs {s.shape}"
            )
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = (
            torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
            * s_zm
            / (l2norm(s_zm, keepdim=True) ** 2 + eps)
        )
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def forward(self, x, y):
        return -self.cal_sisnr(x, y).mean()


class SNRLoss(torch.nn.Module):
    def __init__(self, tao=1e-3):
        super().__init__()
        self.tao = tao

    def l2norm(self, mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    def forward(self, x, s, eps=1e-8):
        if x.shape != s.shape:
            raise RuntimeError(
                f"Dimension mismatch when calculate si-snr, {x.shape} vs {s.shape}"
            )

        loss = 10 * torch.log10(
            self.l2norm(s - x) ** 2 + self.tao * self.l2norm(s) ** 2 + eps
        ) - 10 * torch.log10(self.l2norm(s) ** 2 + eps)
        return loss.mean()


class STOILoss(torch.nn.Module):
    def __init__(self, sr):
        super().__init__()
        self.NegSTOI = NegSTOILoss(sample_rate=sr)

    def forward(self, x, s):
        return self.NegSTOI(x, s).mean()


class STOILevelLoss(torch.nn.Module):
    def __init__(self, sr, alpha, block_size=0.4, overlap=0.7, gamma_a=-70):
        super().__init__()
        self.NegSTOI = NegSTOILoss(sample_rate=sr)
        self.alpha = alpha

        "rms measurement"
        self.frame_size = int(block_size * sr)
        self.frame_shift = int(block_size * sr * (1 - overlap))
        self.unfold = torch.nn.Unfold(
            (1, self.frame_size), stride=(1, self.frame_shift)
        )
        self.gamma_a = gamma_a

        "mse"
        self.cal_mse = torch.nn.MSELoss()

    def measure_loudness(self, signal, eps=1e-8):
        x_unfold = self.unfold(signal.unsqueeze(1).unsqueeze(2))

        z = (
            torch.sum(x_unfold**2, dim=1) / self.frame_size
        )  # mean square for each frame
        el = -0.691 + 10 * torch.log10(z + eps)

        idx_a = torch.where(el > self.gamma_a, 1, 0)
        z_ave_gated_a = torch.sum(z * idx_a, dim=1, keepdim=True) / (
            torch.sum(idx_a, dim=1, keepdim=True) + eps
        )
        gamma_r = -0.691 + 10 * torch.log10(z_ave_gated_a + eps) - 10

        idx_r = torch.where(el > gamma_r, 1, 0)
        idx_a_r = idx_a * idx_r
        z_ave_gated_a_r = torch.sum(z * idx_a_r, dim=1, keepdim=True) / (
            torch.sum(idx_a_r, dim=1, keepdim=True) + eps
        )
        lufs = -0.691 + 10 * torch.log10(z_ave_gated_a_r + eps)  # loudness
        return lufs

    def forward(self, x, s):
        loudness_x = self.measure_loudness(x)
        loudness_s = self.measure_loudness(s)
        LevelLoss = self.alpha * self.cal_mse(loudness_x, loudness_s)
        return LevelLoss + self.NegSTOI(x, s).mean()
