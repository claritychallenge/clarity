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

### Enhancement

The baseline enhancement simply takes the 6-channel hearing aid inputs and reduces this to a stereo hearing aid output by passing through the 'front' microphone signal of the left and right ear.

To run the baseline enhancement system, first, specify `root` in `config.yaml` to point to where you have installed the clarity data. You can also define your own `path.exp_folder` to store enhanced signals and evaluate
Then run:

```bash
python enhance.py
```

Alternatively, you can provide the root variable on the command line, e.g.,

```bash
python enhance.py path.root=/Volumes/data/clarity_CEC2_data
```

The folder `enhanced_signals` will appear in the `exp` folder.

### Evaluation

The `evaluate.py`  will first pass signals through a provided hearing aid amplification stage using a NAL-R [[1](#references)] fitting amplification and a simple automatic gain compressor. The amplification is determined by the audiograms defined by the scene-listener pairs in `clarity_data/metadata/scenes_listeners.dev.json` for the development set. After amplification, the evaluate function calculates the better-ear HASPI  [[2](#references)] and better-ear HASQI  [[3](#references)] scores. The average of these two is computed and returned for each signal.

To run the evaluation stage, make sure that `path.root` is set in the `config.yaml` file and then run

```bash
python evaluate.py
```

The full evaluation set is 7500 scene-listener pairs and will take a long time to run. A standard small set which uses 1/15 of the data has been defined and can be run with

```bash
python evaluate.py evaluate.small_test=True
```

A csv file containing the HASPI, HASQI and combined scores will be generated in the `exp_folder`.

When computing HASPI and HASQI, the `_target_anechoic_CH1.wav` is used as the reference, with its level normalised to match that of the corresponding `_target_CH1.wav`.

### Reporting results

Once the evaluation script has finished running, the final result can be reported with

```bash
python report_score.py
```

Or if you have run the small evaluation

```bash
python report_score.py evaluate.small_test=True
```

The score for the baseline enhancement is 0.185 overall (0.239 HASPI; 0.132 HASQI).

Please note: HASPI and HASQI employ random thresholding noise so you will not get identical scores unless the random seed is set (in the given recipe, the random seed for each signal is set the last eight digits of the scene md5). However, if the seed is not set the differences between runs should be small (order of 1e-6).

## References

* [1] Byrne, Denis, and Harvey Dillon. "The National Acoustic Laboratories'(NAL) new procedure for selecting the gain and frequency response of a hearing aid." Ear and hearing 7.4 (1986): 257-265.
* [2] Kates J M, Arehart K H. The hearing-aid speech perception index (HASPI) J. Speech Communication, 2014, 65: 75-93.
* [3] Kates, J.M. and Arehart, K.H., 2014. "The hearing-aid speech quality index (HASQI) version 2". Journal of the Audio Engineering Society. 62 (3): 99–117.
