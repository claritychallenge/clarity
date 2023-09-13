"""Test the ConvTasNet model."""
import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from clarity.dataset.cec1_dataset import CEC1Dataset
from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet, overlap_and_add


def test_overlap_add(regtest):
    """Test the overlap and add function."""
    signal = torch.Tensor(np.sin(np.arange(0, 2 * np.pi, 0.001) * 10))
    signal = torch.reshape(signal, (2, -1))

    frame_step = 100
    output = overlap_and_add(signal, frame_step, None)
    regtest.write(f"overlap add output: \n{output.detach().numpy()}\n")


def test_convtasnet(regtest):
    """Test the ConvTasNet model."""
    torch.manual_seed(0)
    cfg = {}
    cfg["path"] = {"exp_folder": "./"}
    cfg["mc_convtasnet"] = {
        "N_spec": 256,
        "N_spat": 128,  # 6 * 30
        "L": 20,
        "B": 256,
        "H": 512,
        "P": 3,
        "X": 6,
        "R": 4,
        "C": 1,  # num_speakers
        "num_channels": 6,  # should be consistent with dataloader num_channels
        "norm_type": "cLN",
        "causal": True,
        "mask_nonlinear": "relu",
        "device": "cpu",
    }

    cfg["test_dataset"] = {
        "scenes_folder": "tests/test_data/scenes",
        "scenes_file": "tests/test_data/metadata/scenes.test.json",
        "sample_rate": 44100,
        "downsample_factor": 20,
        "wav_sample_len": None,
        "wav_silence_len": 0,
        "num_channels": 2,
        "norm": 1,
        "testing": False,
    }

    cfg["test_loader"] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,  # Overhead of multiprocessing not worth it for tiny dataset
    }
    cfg = OmegaConf.create(cfg)
    device = "cpu"
    test_set = CEC1Dataset(**cfg.test_dataset)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, **cfg.test_loader)

    if cfg.test_dataset.downsample_factor != 1:
        down_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.test_dataset["sample_rate"],
            new_freq=cfg.test_dataset["sample_rate"]
            // cfg.test_dataset.downsample_factor,
            resampling_method="sinc_interp_hann",
        )
        up_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.test_dataset["sample_rate"]
            // cfg.test_dataset.downsample_factor,
            new_freq=cfg.test_dataset["sample_rate"],
            resampling_method="sinc_interp_hann",
        )

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="testing"):
            noisy, _scene = batch
            out = []
            for _ear in ["left", "right"]:
                torch.cuda.empty_cache()
                # load denoising module
                den_model = ConvTasNet(**cfg.mc_convtasnet)

                den_model = torch.nn.parallel.DataParallel(den_model.to(device))
                den_model.eval()

                noisy = torch.reshape(noisy, (1, 6, -1))
                noisy = noisy.to(device).cpu()
                if cfg.test_dataset["downsample_factor"] != 1:
                    proc = down_sample(noisy)
                enhanced = (den_model(proc)).squeeze(1)
                enhanced = enhanced.cpu()
                if cfg.test_dataset["downsample_factor"] != 1:
                    enhanced = up_sample(enhanced)
                enhanced = torch.clamp(enhanced, -1, 1)
                out.append(enhanced.detach().cpu().numpy()[0])

            out = np.stack(out, axis=0).transpose()
    regtest.write(f"ctn output: \n{np.round(out, 6)}\n")
