# The 3rd Clarity Enhancement Challenge (CEC3)

Clarity challenge code for the 3rd Clarity Enhancement Challenge.

For more information please visit the [challenge website](https://claritychallenge.org/docs/cec3/cec3_intro).

Clarity tutorials are [now available](https://claritychallenge.github.io/clarity_CC_doc/tutorials). The tutorials introduce the Clarity installation, how to interact with Clarity metadata, and also provide examples of baseline systems and evaluation tools.

## Data structure

The 3rd Clarity Enhancement Challenge consists of three separate tasks each with its own training and evaluation data. Details for how to obtain the data can be found on the [challenge website](https://claritychallenge.org/docs/cec3/cec3_data).

The data is distributed as one or more separate packages in `tar.gz` format.

Unpack all packages under the same root directory using

```bash
tar -xvzf <PACKAGE_NAME>
```

The initially released data is in the package `clarity_CEC3_data.v1_0.tar.gz` and has the following structure:

```text
clarity_CEC3_data
|── manifest
|── task1
|   |── clarity_data
|   |   |── dev
|   |   |   |── scenes
|   |   |   └── speaker_adapt
|   |   |── metadata
|   |   └── train
|   └── hrir
|       └── HRIRs_MAT
|── task2
|   └── clarity_data
|       |── dev
|       |   |── interferers
|       |   |── scenes
|       |   |── speaker_adapt
|       |   └── targets
|       |── metadata
|       └── train
|           |── interferers
|           |── scenes
|           └── targets
└── task3


```

## Baseline

In the `baseline/' folder, we provide code for running the baseline enhancement system and performing the objective evaluation. The same system can be used for all three tasks by setting the configuration appropriately.

The scripts are controlled by three variables.

- `task` - The task to evaluate. This can be `task1`, `task2` or `task3`.
- `path.root` - The root directory where you clarity data is stored.
- `path.exp` - A directory that will be used to store intermediate files and the final evaluation results.

These can be set in the `config.yaml` file or provided on the command line. In the following they are being set on the command line.

### Enhancement

The baseline enhancement simply takes the 6-channel hearing aid inputs and reduces this to a stereo hearing aid output by passing through the 'front' microphone signal of the left and right ear.

The stereo pair is then passed through a provided hearing aid amplification stage using a NAL-R [[1](#references)] fitting amplification and a simple automatic gain compressor. The amplification is determined by the audiograms defined by the scene-listener pairs in `clarity_data/metadata/scenes_listeners.dev.json` for the development set. After amplification, the evaluate function calculates the better-ear HASPI  [[2](#references)].

To run the baseline enhancement system use, first set the `task`, `path.root` and `path.exp` variables in the `config.yaml` file and then run,

```bash
python enhance.py
```

Alternatively, you can provide the task and paths on the command line, e.g.,

```bash
python enhance.py task=task1 path.root=/Users/jon/clarity_CEC3_data path.exp=/Users/jon/exp
```

Where `/Users/jon` is replaced with the path to the root of the clarity data and the experiment folder.

The folders `enhanced_signals`  and `amplified_signals` will appear in the `exp` folder. Note, the experiment folder will be created if it does not already exist.

### Evaluation

The evaluate script computes the HASPI scores for the signals stored in the `amplified_signals` folder. The script will read the scene-listener pairs from the development set and calculate the HASPI score for each pair. The final score is the mean HASPI score across all pairs. It can be run as,

```bash
python evaluate.py task=task1 path.root=/Users/jon/clarity_CEC3_data path.exp=/Users/jon/exp

```

The full evaluation set is 7500 scene-listener pairs and will take a long time to run, i.e., around 8 hours on a MacBook Pro. A standard small set which uses 1/15 of the data has been defined. This takes around 30 minutes to evaluate and can be run with,

```bash
python evaluate.py task=task1 path.root=/Users/jon/clarity_CEC3_data path.exp=/Users/jon/exp evaluate.small_test=True
```

Alternatively, see the section below, 'Running with multiple threads', for how to run with multiple threads or on an HPC system.

The evaluation script will generate a CSV file containing the HASPI scores for each sample. This can be found in `<path.exp>/scores`

### Reporting results

Once the evaluation script has finished running, the final result can be reported with

```bash
python report_score.py task=task1 path.root=/Users/jon/clarity_CEC3_data path.exp=/Users/jon/exp
```

Or if you have run the small evaluation

```bash
python report_score.py task=task1 path.root=/Users/jon/clarity_CEC3_data path.exp=/Users/jon/exp evaluate.small_test=True
```

The scores for Task 1 and Task 2 should be as follows.

Task 1

```text
Evaluation set size: 7500
Mean HASPI score: 0.22178678134846783

                 SNR     haspi
SNR
(-12, -9] -10.498088  0.052545
(-9, -6]   -7.541468  0.080589
(-6, -3]   -4.477046  0.143096
(-3, 0]    -1.432494  0.239527
(0, 3]      1.470118  0.352110
(3, 6]      4.492380  0.477001
```

Task 2

```text
Evaluation set size: 7500
Mean HASPI score: 0.18643217215546573

                 SNR     haspi
SNR
(-12, -9] -10.545927  0.034330
(-9, -6]   -7.552687  0.055647
(-6, -3]   -4.538335  0.096237
(-3, 0]    -1.455963  0.178413
(0, 3]      1.434074  0.296364
(3, 6]      4.507484  0.432177
```

## Tips

### Configuring with Hydra

The code is using [Hydra](https://hydra.cc) for configuration management. This allows for easy configuration of the system. The configuration file is `config.yaml` in the `baseline` folder. The task, root and exp variables can be set in this file to avoid having to set them on every command line. Simply replace the `???` entries with the appropriate values.

You can make alternative configurations and store them in separate `yaml` files. These can then be used to override the default configuration, e.g.,

```bash
python enhance.py python report_score.py --config-name my_task1_config.yaml
```

You can get help on any of the commands with

```bash
python enhance.py --help
```

And specific help on Hydra usage with

```bash
python enhance.py --hydra-help
```

### Running with multiple threads

The `evaluate.py` script can be sped up by running with multiple processes, i.e. each process will evaluate a separate block of scenes and generate its own csv file. The `report_score.py` script will then combine these csv files to produce a single result.

To do this we can use the Hydra `--multirun` flag and set multiple values for `evaluate.first_scene`. For example, to run with 4 threads we can split the 7500 scenes into 4 blocks of 1875 scenes each and run with,

```bash
python evaluate.py evaluate.first_scene="0,1875,3750,5625" evaluate.n_scenes=1875 --multirun
```

Hydra has a Python like system for specifying ranges, so the above command is equivalent to

```bash
python evaluate.py  evaluate.first_scene="range(0,7500,1875) evaluate.n_scenes=1875 --multirun
```

If we wanted to split into jobs with just 100 scenes per job we could use

```bash
python evaluate.py evaluate.first_scene="range(0,7500,100)" evaluate.n_scenes=500 --multirun
```

Hydra will launch these job using configuration that can be found in `hydra/launcher/cec3_submitit_local.yaml`.

The same approach can be used to run jobs on a SLURM cluster using configuration in `hydra/launcher/cec3_submitit_slurm.yaml`.

```bash
python evaluate.py hydra/launcher=cec3_submitit_slurm evaluate.first_scene="range(0,7500,100)" evaluate.n_scenes=100 --multirun
```

!!!Note In the examples above it is assumed that the `task`, `path.root` and `path.exp` variables are set in the `config.yaml` file.

!!!Note Hydra has plugin support for other job launchers. See the [Hydra documentation for more information](https://hydra.cc/docs/intro/).

## References

- [1] Byrne, Denis, and Harvey Dillon. "The National Acoustic Laboratories'(NAL) new procedure for selecting the gain and frequency response of a hearing aid." Ear and hearing 7.4 (1986): 257-265.
- [2] Kates J M, Arehart K H. The hearing-aid speech perception index (HASPI) J. Speech Communication, 2014, 65: 75-93.
