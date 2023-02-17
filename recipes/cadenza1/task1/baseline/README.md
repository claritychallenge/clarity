# The First Cadenza Challenge (CAD1)

Cadenza challenge code for the First Cadenza Challenge (CAD1) Task1 and Task2

For more information please visit the [challenge website](https://cadenzachallenge.org/docs/cadenza1/cc1_intro).

## Data structure

The First Cadenza Challenge task 1 uses the MUSDB18 dataset.
The First Cadenza Challenge task 1 uses the MUSDB18 dataset.
To download data, please visit [here]().
Demo data can be downloaded from [here](https://drive.google.com/file/d/1ZQ9Z1Z4Z7Z4Z7Z4Z7Z4Z7Z4Z7Z4Z7Z4Z/view?usp=sharing).

Unpack packages under the same root directory using

```bash
tar -xvzf <PACKAGE_NAME>
```

## Baseline

The baseline system consists of a music decomposition into
`vocals`, `drums`, `bass` and `others` (VDBO) stems
using [Hybrid Demucs](https://github.com/facebookresearch/demucs)[1].

Then, a simple NAL-R [2] fitting amplification and a simple automatic gain compressor is
applied to each VDBO stem.

The final remixed audio is obtained by summing the amplified VDBO stems.

### Baseline enhancement

To run the baseline enhancement system, firstly specify `root` in config.yaml. You can also define your own `path.exp_folder` to store enhanced signals and evaluated results.
The baseline will generate a left and right signal for each VDBO stem, totalling 8 stems.
Then run:
```bash
python enhance.py
```
The folder `enhanced_signals` will appear in the `exp` folder.


### Evaluation

The `evaluate.py` calculates the HAAQI [3] score for each left and right VDBO stem given the
song-listener pairs for the development set.
Specify `path.exp_folder` to store the results. Then run:
```bash
python evaluate.py
```
A csv file containing the eight HAAQI scores and an average
will be generated in the `exp_folder`.

To check the HAAQI code, see [here](../../clarity/evaluator/haaqi).

Please note: you will not get identical HAAQI scores for the same signals if the random seed is not determined
(in the given recipe, the random seed for each signal is set the last eight digits of the song md5).
As there are random noises generated within HAAQI, but the differences should be sufficiently small.
We ran evaluation for the baseline for five times, and the average overall score is XXX +/- XXXX.

## References

* [1] Défossez, A. "Hybrid Spectrogram and Waveform Source Separation". Proceedings of the ISMIR 2021 Workshop on Music Source Separation.
* [2] Byrne, Denis, and Harvey Dillon. "The National Acoustic Laboratories'(NAL) new procedure for selecting the gain and frequency response of a hearing aid." Ear and hearing 7.4 (1986): 257-265.
* [3] Kates J M, Arehart K H. "The Hearing-Aid Audio Quality Index (HAAQI)". IEEE/ACM transactions on audio, speech, and language processing, 24(2), 354–365.
