from pathlib import Path

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
    exp_folder = Path(cfg.path.exp_folder)
    output_folder = exp_folder / f"enhanced_{cfg.listener.id}"
    output_folder.mkdir(parents=True, exist_ok=True)

    test_set = CEC1Dataset(**cfg.test_dataset)
    test_loader = DataLoader(dataset=test_set, **cfg.test_loader)

    down_sample = up_sample = None
    if cfg.downsample_factor != 1:
        down_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sample_rate,
            new_freq=cfg.sample_rate // cfg.downsample_factor,
            resampling_method="sinc_interp_hann",
        )
        up_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sample_rate // cfg.downsample_factor,
            new_freq=cfg.sample_rate,
            resampling_method="sinc_interp_hann",
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
                den_model_path = exp_folder / f"{ear}_den/best_model.pth"

                den_model.load_state_dict(
                    torch.load(den_model_path, map_location=device)
                )
                _den_model = torch.nn.parallel.DataParallel(den_model.to(device))
                _den_model.eval()

                # load amplification module
                amp_model = AudiometricFIR(**cfg.fir)
                amp_model_path = exp_folder / f"{ear}_amp/best_model.pth"

                amp_model.load_state_dict(
                    torch.load(amp_model_path, map_location=device)
                )
                _amp_model = torch.nn.parallel.DataParallel(amp_model.to(device))
                _amp_model.eval()

                noisy = noisy.to(device)
                proc = noisy
                if down_sample is not None:
                    proc = down_sample(noisy)
                enhanced = amp_model(den_model(proc)).squeeze(1)
                if up_sample is not None:
                    enhanced = up_sample(enhanced)
                enhanced = torch.clamp(enhanced, -1, 1)
                out.append(enhanced.detach().cpu().numpy()[0])

            out = np.stack(out, axis=0).transpose()
            write(
                output_folder / f"{scene[0]}_{cfg.listener.id}_HA-output.wav",
                out,
                cfg.sample_rate,
            )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()
