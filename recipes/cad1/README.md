# The 1st Cadenza Challenge
Cadenza challenge code for the 1st challenge.

For more information about the Cadenza Challenge please visit https://cadenzachallenge.com

Cadenza tutorials are now available in https://cadenzachallenge.com.
The tutorials introduce the Cadenza installation, how to interact with Cadenza metadata,
and also provide examples of baseline systems and evaluation tools.

## Data structure

To download data, please visit [here](https://mab.to/zU7TS8jJelkoD).
The data is split into three packages:
`clarity_CEC2_core.v1_0.tgz` [28 GB],
`clarity_CEC2_train.v1_0.tgz` [69 GB] and
`clarity_CEC2_hoairs.v1_0.tgz` [144 GB].

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

