import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from asteroid.engine.system import System
from local import (
    Compose,
    ConvTasNetStereo,
    RebalanceMusicDataset,
    augment_channelswap,
    augment_gain,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)


def main(conf):
    source_augmentations = Compose([augment_gain, augment_channelswap])

    dataset_kwargs = {
        "root_path": Path(conf["data"]["root_path"]),
        "sample_rate": conf["data"]["sample_rate"],
        "target": conf["data"]["target"],
    }

    train_set = RebalanceMusicDataset(
        split="train",
        music_tracks_file=f"{conf['data']['music_tracks_file']}/music.train.json",
        samples_per_track=conf["data"]["samples_per_track"],
        segment_length=conf["data"]["segment_length"],
        random_segments=True,
        random_track_mix=True,
        source_augmentations=source_augmentations,
        **dataset_kwargs,
    )

    val_set = RebalanceMusicDataset(
        music_tracks_file=f"{conf['data']['music_tracks_file']}/music.valid.json",
        split="valid",
        segment_length=None,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = ConvTasNetStereo(
        **conf["convtasnet"], samplerate=conf["data"]["sample_rate"]
    )

    optimizer = torch.optim.Adam(model.parameters(), conf["optim"]["lr"])

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = torch.nn.L1Loss()
    # loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=1, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=True)
        )

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        devices="auto",
        gradient_clip_val=5.0,
        accumulate_grad_batches=conf["training"]["aggregate"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    from pprint import pprint as print

    import yaml
    from asteroid.utils import parse_args_as_dict, prepare_parser_from_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
