# The 2nd Clarity Enhancement Challenge (CEC2)
Clarity challenge code for the 2nd enhancement challenge (CEC2).

For more information about the Clarity Challenge please visit https://claritychallenge.github.io/clarity_CC_doc/

## Data structure

To download data, please visit [here](https://mab.to/zU7TS8jJelkoD). The data is split into three packages: `clarity_CEC2_core.v1_0.tgz` [28 GB], `clarity_CEC2_train.v1_0.tgz` [69 GB] and `clarity_CEC2_hoairs.v1_0.tgz` [144 GB].

Unpack packages under the same root directory using

```bash
tar -xvzf <PACKAGE_NAME>
```

**Core** contains metadata and development set signals, which can be used for validate existing systems

```text
clarity_data
|   hrir/HRIRs_MAT 167M
|
└───dev
|   └───rooms
|   |   |   ac 20M
|   |   |   rpf 79M
|   |
|   └───interferers
|   |   |   music 5.8G
|   |   |   noise 587M
|   |   |   speech 1.4G
|   |
|   └───scenes 39G
|   |
|   └───targets 1.3G
|   |
|   └───speaker_adapt 20M
|   
└───metadata
    |   scenes.train.json
    |   scenes.dev.json
    |   rooms.train.json
    |   rooms.dev.json
    |   masker_music_list.json
    |   masker_nonspeech_list.json
    |   masker_speech_list.json
    |   target_speech_list.json
    |   hrir_data.json
    |   listeners.json
    |   scenes_listeners.dev.json
    |   ...

```

**Train** contains training set, which can be used to optimise a system

```text
clarity_data
└───train
    └───rooms
    |   |   ac 48M
    |   |   rpf 190M
    |
    └───interferers
    |   |   music 16GG
    |   |   noise 3.9M
    |   |   speech 4.5G
    |
    └───scenes 89G
    |
    └───targets 2.8G

```

**HOA_IRs** contains impulse responses for reproducing the scenes or for rendering more training data (scenes).

```text
clarity_data
└───train/rooms/HOA_IRs 117G
|
└───dev/rooms/HOA_IRs 49G
```

## Data preparation

In this folder, we provide the code used for building scenes (i.e., generating `metadata/scenes.train.json` and `metadata/scenes.dev.json`), and rendering the scene signals dependent on the parameters in the json files. You can reproduce all the scenes with `HOA_IRs`.

To reproduce the training and development set, first specify the `path.root` in `config.yaml`, i.e., replace `???` with `YOUR_DATA_PATH/clarity_CEC2_data`.

Second run (will skip if the json files of _rooms_ or _scenes_ are already in `metadata/`)

```bash
python build_scenes.py
```

Third run the `render_scenes.py`.

If single-run locally:

```bash
python render_scenes.py
```

The Hydra submitit feature is used for parallel multi-run, see [https://hydra.cc/docs/plugins/submitit_launcher/]

If multi-run locally, make sure the `override hydra/launcher` is set `cec2_submitit_local`, and specify the parameters in `hydra/launcher/cec2_submitit_local.yaml`, then:

```bash
# 50 subjobs
python render_scenes.py 'render_starting_chunk=range(0, 500, 10)' --multirun
```

If multi-run on a Slurm cluster, make sure the `override hydra/launcher` is set `cec2_submitit_slurm`, and specify the parameters in `hydra/launcher/cec2_submitit_slurm.yaml`, then submit the `render_scenes.sh`:

```bash
sbatch render_scenes.sh
```

## Baseline

In this folder, we provide code for generating more training data, objective evaluation with binaural HASPI [1]. The baseline system will be released soon.

### Data generation

The method of using this is the same as for `Data preparation`. A `scenes.train_additional.json` will be generated in the `clarity_data/metadata/`, and additional training data will be generated in `clarity_data/train/additional_scenes/`.

### Evaluation

The `evaluate.py` calculates the HASPI score given the scene-listener pairs in `clarity_data/metadata/scenes_listeners.dev.json` for the development set. Specify `path.exp_folder` to store the results. To check the HASPI code, see [here](../../clarity/evaluator/haspi).

## References

* [1] Kates J M, Arehart K H. The hearing-aid speech perception index (HASPI)[J]. Speech Communication, 2014, 65: 75-93.

