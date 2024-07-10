import argparse
import os
import sys
from pathlib import Path

import musdb
import museval
import soundfile as sf
import torch
import yaml
from local import ConvTasNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex",
    type=int,
    default=10,
    help="Number of audio examples to save, -1 means all",
)

compute_metrics = ["si_sdr", "sdr", "sir", "sar"]


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")

    model = ConvTasNet(
        **conf["train_conf"]["convtasnet"],
        samplerate=conf["train_conf"]["data"]["sample_rate"],
    )

    saved = torch.load(model_path, map_location="cpu")
    model.load_state_dict(saved["state_dict"])

    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()

    model_device = next(model.parameters()).device

    # Evaluation is mode using 'remix' mixture
    test_set = musdb.DB(
        root=conf["train_conf"]["data"]["root"], subsets="test", is_wav=True
    )
    results = museval.EvalStore()

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    Path(eval_save_dir).mkdir(exist_ok=True, parents=True)

    txtout = os.path.join(eval_save_dir, "results.txt")
    fp = open(txtout, "w")

    torch.no_grad().__enter__()
    for track in test_set:
        input_file = os.path.join(
            conf["train_conf"]["data"]["root"], "test", track.name, "mixture.wav"
        )
        # Forward the network on the mixture.
        mix, rate = sf.read(input_file, always_2d=True, start=0, stop=None)

        # Separate
        mix = torch.tensor(mix.T, dtype=torch.float).to(model_device)

        est_sources = model.forward(mix.unsqueeze(0))
        est_sources = est_sources.squeeze(0).cpu().data.numpy()

        estimates = {}
        estimates["vocals"] = est_sources[0].T
        estimates["accompaniment"] = est_sources[1].T

        output_path = Path(os.path.join(eval_save_dir, track.name))
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"Processing... {track.name}", file=sys.stderr)
        print(track.name, file=fp)

        for target, estimate in estimates.items():
            sf.write(
                str(output_path / Path(target).with_suffix(".wav")),
                estimate,
                conf["train_conf"]["data"]["sample_rate"],
            )
        track_scores = museval.eval_mus_track(track, estimates)
        results.add_track(track_scores.df)
        print(track_scores, file=sys.stderr)
        print(track_scores, file=fp)
    print(results, file=sys.stderr)
    print(results, file=fp)
    results.save(os.path.join(eval_save_dir, "results.pandas"))
    results.frames_agg = "mean"
    print(results, file=sys.stderr)
    print(results, file=fp)
    fp.close()


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)

    print("Done!")
