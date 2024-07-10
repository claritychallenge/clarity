import argparse
import os
import sys
from pathlib import Path

import museval
import soundfile as sf
import torch
import yaml
from local import ConvTasNet, RebalanceMusicDataset
from museval import TrackStore, evaluate
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir",
    default="out",
    type=str,
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
    target_inst = conf["train_conf"]["data"]["target"]
    dataset_kwargs = {
        "root_path": Path(conf["train_conf"]["data"]["root_path"]),
        "sample_rate": conf["train_conf"]["data"]["sample_rate"],
        "target": target_inst,
    }

    test_set = RebalanceMusicDataset(
        split="test",
        music_tracks_file=(
            f"{conf['train_conf']['data']['music_tracks_file']}/music.eval.json"
        ),
        **dataset_kwargs,
        samples_per_track=1,
        random_segments=False,
        random_track_mix=False,
    )

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    Path(eval_save_dir).mkdir(exist_ok=True, parents=True)

    txtout = os.path.join(eval_save_dir, "results.txt")
    fp = open(txtout, "w")

    results = museval.EvalStore(frames_agg="median", tracks_agg="median")

    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = test_set[idx]
        mix = mix.to(model_device)
        est_sources = model.forward(mix.unsqueeze(0))
        sources_np = sources.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()

        estimates = {}
        estimates[target_inst] = est_sources_np[0].T
        estimates["accompaniment"] = est_sources_np[1].T

        references = {}
        references[target_inst] = sources_np[0].T
        references["accompaniment"] = sources_np[1].T

        output_path = Path(os.path.join(eval_save_dir, test_set.track_name))
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"Processing... {test_set.track_name}", file=sys.stderr)
        print(test_set.track_name, file=fp)

        for target, estimate in estimates.items():
            sf.write(
                str(output_path / Path(target).with_suffix(".wav")),
                estimate,
                conf["train_conf"]["data"]["sample_rate"],
            )
        # Evaluate using museval
        track_scores = eval_track(
            references,
            estimates,
            eval_targets=[target_inst, "accompaniment"],
            track_name=test_set.track_name,
            sample_rate=conf["sample_rate"],
        )

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


def eval_track(
    reference,
    user_estimates,
    eval_targets=[],
    track_name="",
    mode="v4",
    win=1.0,
    hop=1.0,
    sample_rate=44100,
):
    audio_estimates = []
    audio_reference = []

    data = TrackStore(win=win, hop=hop, track_name=track_name)

    if len(eval_targets) >= 2:
        # compute evaluation of remaining targets
        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(reference[target])

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win * sample_rate),
            hop=int(hop * sample_rate),
            mode=mode,
        )

        # iterate over all evaluation results except for vocals
        for i, target in enumerate(eval_targets):
            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist(),
            }

            data.add_target(target_name=target, values=values)

    return data


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
