import os

import hydra
import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig
from soundfile import write
from torch.utils.data import DataLoader
from tqdm import tqdm

from clarity.dataset.cec1_dataset import CEC1Dataset
from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet
from clarity.enhancer.dsp.filter import AudiometricFIR


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    output_folder = os.path.join(cfg.path.exp_folder, "enhanced_" + cfg.listener.id)
    os.makedirs(output_folder, exist_ok=True)

    test_set = CEC1Dataset(**cfg.test_dataset)
    test_loader = DataLoader(dataset=test_set, **cfg.test_loader)

    if cfg.downsample_factor != 1:
        down_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sr,
            new_freq=cfg.sr // cfg.downsample_factor,
            resampling_method="sinc_interpolation",
        )
        up_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sr // cfg.downsample_factor,
            new_freq=cfg.sr,
            resampling_method="sinc_interpolation",
        )

    device = "cuda" if torch.cuda.is_available() else None

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="testing"):
            noisy, scene = batch
            out = []
            for ear in ["left", "right"]:
                torch.cuda.empty_cache()
                # load denoising module
                den_model = ConvTasNet(**cfg.mc_conv_tasnet)
                den_model_path = os.path.join(
                    os.path.join(cfg.path.exp_folder, ear + "_den"), "best_model.pth"
                )
                den_model.load_state_dict(
                    torch.load(den_model_path, map_location=device)
                )
                _den_model = torch.nn.parallel.DataParallel(den_model.to(device))
                _den_model.eval()

                # load amplification module
                amp_model = AudiometricFIR(**cfg.fir)
                amp_model_path = os.path.join(
                    os.path.join(cfg.path.exp_folder, ear + "_amp"), "best_model.pth"
                )
                amp_model.load_state_dict(
                    torch.load(amp_model_path, map_location=device)
                )
                _amp_model = torch.nn.parallel.DataParallel(amp_model.to(device))
                _amp_model.eval()

                noisy = noisy.to(device)
                if cfg.downsample_factor != 1:
                    proc = down_sample(noisy)
                enhanced = amp_model(den_model(proc)).squeeze(1)
                if cfg.downsample_factor != 1:
                    enhanced = up_sample(enhanced)
                enhanced = torch.clamp(enhanced, -1, 1)
                out.append(enhanced.detach().cpu().numpy()[0])

            out = np.stack(out, axis=0).transpose()
            write(
                os.path.join(
                    output_folder,
                    scene[0] + "_" + cfg.listener.id + "_" + "HA-output.wav",
                ),
                out,
                cfg.sr,
            )


if __name__ == "__main__":
    run()
