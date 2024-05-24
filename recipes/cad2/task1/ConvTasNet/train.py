"""
Script to train causal and noncausal ConvTasNet on MUSDB18.
Model is trained to separate the vocals from the background music.
"""

import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from asteroid.engine.system import System
from local import Compose, ConvTasNet, MUSDB18Dataset, augment_channelswap, augment_gain
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

validation_tracks = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)


def create_datasets_and_loaders(conf):
    source_augmentations = Compose([augment_gain, augment_channelswap])

    dataset_kwargs = {
        "root": Path(conf["data"]["root"]),
        "mix_background": conf["data"]["mix_background"],
        "sample_rate": conf["data"]["sample_rate"],
    }

    train_set = MUSDB18Dataset(
        split="train",
        random_segments=True,
        random_track_mix=True,
        exclude_tracks=validation_tracks,
        source_augmentations=source_augmentations,
        segment=conf["data"]["segment"],
        samples_per_track=conf["data"]["samples_per_track"],
        **dataset_kwargs,
    )

    val_set = MUSDB18Dataset(
        split="train",
        subset=validation_tracks,
        segment=None,
        samples_per_track=1,
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
        batch_size=1,
        num_workers=conf["training"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, train_set, val_loader, val_set


def create_model_and_optimizer(conf):
    model = ConvTasNet(**conf["convtasnet"], samplerate=conf["data"]["sample_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["optim"]["lr"])
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    return model, optimizer, scheduler


def create_trainer(conf, callbacks):
    return pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=conf["main_args"]["exp_dir"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        devices="auto",
        gradient_clip_val=5.0,
        accumulate_grad_batches=conf["training"]["aggregate"],
    )


def get_loss_func():
    return torch.nn.L1Loss()


def create_callbacks(conf, exp_dir):
    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=10, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=True)
        )
    return callbacks, checkpoint


def save_best_model(system, checkpoint, exp_dir, train_set):
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(conf):
    train_loader, train_set, val_loader, _ = create_datasets_and_loaders(conf)
    model, optimizer, scheduler = create_model_and_optimizer(conf)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = get_loss_func()
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
    callbacks, checkpoint = create_callbacks(conf, exp_dir)
    trainer = create_trainer(conf, callbacks)
    trainer.fit(system)

    save_best_model(system, checkpoint, exp_dir, train_set)


if __name__ == "__main__":
    from pprint import pprint as print

    from asteroid.utils import parse_args_as_dict, prepare_parser_from_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    def_conf = load_config("local/conf.yml")
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
