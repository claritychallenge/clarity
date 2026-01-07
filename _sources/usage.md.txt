# Usage

Before getting started you should ensure you have installed pyClarity within a virtual environment, if you've not
already done so please refer to the [installation instructions](installation.md).

To check run `pip show pyclarity` and you should get some information about the installed version.

``` bash
Version: 0.1.0
Summary: Tools for the Clarity Challenge
Home-page: https://github.com/claritychallenge/clarity
Author: The PyClarity team
Author-email: clarity-group@sheffield.ac.uk
License: MIT
Location: /home/usera/work/projects/claritychallenge/clarity
Requires: audioread, hydra-core, hydra-submitit-launcher, importlib-metadata, librosa, matplotlib, numpy, omegaconf, pandas, pyloudnorm, scikit-learn, scipy, SoundFile, tqdm
Required-by:
```

If you don't see similar to the above then check that you have activated the Virtual Environment you made the install
under, if the package still isn't found then you should go through the installation process again within your Virtual
Environment.

## Jupyter Notebooks

These tutorials are available as Jupyter Notebooks [pyClarity Tutorials](https://claritychallenge.org/tutorials) that
run in Google CoLab.

The examples and code below take you through the using the pyClarity tools using smaller demo datasets which are
provided under `clarity.data.demo_data` and have specific functions for loading.

## 01 Installing pyClarity and Using Metadata

This demonstration uses only the `metadata` datasets and it is downloaded to the `clarity_data/demo/metadata/` directory.

``` python
from clarity.data import demo_data

demo_data.get_metadata_demo()
```

This will have created a directory called `clarity_data` containing the metadata files that have been downloaded.

### The structure of the metadata files

There are four metadata files

- `rooms` - geometry of the rooms used for the simulations
- `scenes` - information about the sound scene that is playing in the room
- `listeners` - audiometric data for the hearing-impaired listeners who will listen to the scenes
- `scenes_listeners` - a mapping assigning specific listeners to specific scenes (in the evaluation, each scene will be listened to by three separate listeners)

Information about *individual* rooms, scenes, listeners etc is stored as a [dictionary](https://www.tutorialspoint.com/python/python_dictionary.htm). The complete collections are then stored as either a [list](https://www.tutorialspoint.com/python/python_lists.htm) or dict depending on how the collection is mostly conveniently indexed. The datastructure of the four datatypes is summarized below.

| Dataset            | Structure     | Index         |
|--------------------|---------------|---------------|
| `rooms`            | list of dicts | `int`         |
| `scenes`           | list of dicts | `int`         |
| `listener`         | dict of dicts | `LISTENER_ID` |
| `scenes_listeners` | dict of lists | `LISTENED_ID` |

Data is stored in [JavaScript Object Notation (JSON) format](https://en.wikipedia.org/wiki/JSON) and the components
`scenes`, `rooms`, `listeners` and `scene_listeners` can be loaded with the following.

``` python
import json

with open("clarity_data/demo/metadata/scenes.demo.json") as f:
    scenes = json.load(f)

with open("clarity_data/demo/metadata/rooms.demo.json") as f:
    rooms = json.load(f)

with open("clarity_data/demo/metadata/listeners.json") as f:
    listeners = json.load(f)

with open("clarity_data/demo/metadata/scenes_listeners.dev.json") as f:
    scenes_listeners = json.load(f)
```

Elements of a list are accessed using the numerical index (starting at `0`). Whilst elements of a dictionary are
accessed using the keys. We extract the first (`0`th) scene and inspect the `SNR`

``` python
scene_0 = scenes[0]
print(f"Keys for scene_0         : {scene_0.keys()}")
print(f'Value of SNR for scene_0 : {scene_0["SNR"]}')
# Directly...
print(f'Value of SNR for scene_0 : {scenes[0]["SNR"]}')
```

### Processing Collections of Scenes

Processes can be run over the complete list of scenes using standard Python iteration tools such as `for` and in
particular [list](https://realpython.com/list-comprehension-python/) and [dictionary
comprehension](https://realpython.com/iterate-through-dictionary-python/#using-comprehensions).

``` python
import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(16,9))

# Get list of SNRs of scenes
snr_values = np.array([s["SNR"] for s in scenes], dtype="float32")

# Plot histogram
ax[0].hist(snr_values)
ax[0].set_title("Histogram of SNR values")
ax[0].set_xlabel("SNR (dB)")

# Get list of number of interferers in scenes
n_interferers = np.array([len(s["interferers"]) for s in scenes], dtype="int32")

# Prepare data for boxplot
snr_comparison_data = [
    [s for s, n in zip(snr_values, n_interferers) if n == 2],
    [s for s, n in zip(snr_values, n_interferers) if n == 3],
]

# Plot boxplot
ax[1].boxplot(np.array(snr_comparison_data, dtype="object"))
ax[1].set_xlabel("Number of interferers")
ax[1].set_ylabel("SNR (dB)")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

fig.show()
```

### Associations between metadata types

There are various associations between the metadata types which sometime require cross referencing from one collection
to another.

For example, room dimensions are stored in the room dict rather than directly in the scene dict. So to get the room
dimensions for a given scene, you need to first look at the room ID field in the scene to find the correct room.

One approach to doing this is shown below.

``` python
room_id = scene_0["room"]
# Iterate through rooms to find the one named `room_id`
room = next((item for item in rooms if item["name"] == room_id), None)
print(room["dimensions"])

```

This approach uses a linear search and is therefore not very efficient. If you are going to be doing this often you
might want to convert the list of rooms into a dictionary indexed by room ID, e.g.

``` python
room_dict = {room["name"]: room for room in rooms}
```

You can now look up the dimensions of a scene's room more efficiently,

``` python
room_id = scene_0["room"]
room_dict[room_id]
print(room["dimensions"])
```

### Example: Locating information about the scene's listener

We will now use these ideas to plot the audiograms of one of the listeners associated with a specific scene. The code also prints out some information about the target and listener locations that are stored in the scene's associated room dict.

``` python
scene_no = 32  # this is just an arbitrary index. try any from 0 - 49

scene = scenes[scene_no]

room = room_dict[scene["room"]]
current_listeners = scenes_listeners[scene["scene"]]


print(
    f'\nScene number {scene_no}  (ID {scene["scene"]}) has room dimensions of {room["dimensions"]}'
)

print(
    f'\nSimulated listeners for scene {scene_no} have spatial attributes: \n{room["listener"]}'
)

print(f'\nAudiograms for listeners in Scene ID {scene["scene"]}')


fig, ax = plt.subplots(1, len(current_listeners))

ax[0].set_ylabel("Hearing level (dB)")
for i, l in enumerate(current_listeners):
    listener_data = listeners[l]
    (left_ag,) = ax[i].plot(
        listener_data["audiogram_cfs"],
        -np.array(listener_data["audiogram_levels_l"]),
        label="left audiogram",
    )
    (right_ag,) = ax[i].plot(
        listener_data["audiogram_cfs"],
        -np.array(listener_data["audiogram_levels_r"]),
        label="right audiogram",
    )
    ax[i].set_title(f"Listener {l}")
    ax[i].set_xlabel("Hz")
    ax[i].set_ylim([-100, 10])

plt.legend(handles=[left_ag, right_ag])
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
```

## 02 Running the CEC2 Baseline from the command line

Python comes with a *Read, Eval, Print, Loop* (REPL) interactive shell that can be started from within your Virtual
Environment by typing `python`. Many users prefer the improved [iPython](https://ipython.org/index.html) shell which can
be installed with `pip install ipython` and invoked with `ipython`. Either shell works with the following.

### Install Demo Data

In a shell navigate to the location where you have cloned the pyClarity repository and start an iPython shell

``` bash
$ cd ~/path/to/where/pyclarity/is/cloned
$ BASE_DIR=$(pwd)
$ ipython
Python 3.10.5 (main, Jun  6 2022, 18:49:26) [GCC 12.1.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]:
```

You can now get the demo data with...

``` python
from clarity.data import demo_data

demo_data.get_metadata_demo()
demo_data.get_scenes_demo()
exit
```

Now you have exited to your shell you can change working directory to the location of the shell scripts you wish to run
and see what files have been downloaded.

``` bash
cd clarity/recipes/cec2/baseline
pwd
ls -lha
```

## Inspecting Existing Configuration

All of the included shell scripts take configurable variables from the yaml files in the same directory as the shell script.Typically these are named <code>config.yaml</code>, however, other names may be used if more than one shell script is in a directory.

We can inspect the contents of the config file using <code>!cat</code>:

``` bash
cat config.yaml
```

The general organisation of the config files is hierarchical, with property labels depending on the script in
question. The config file for the enhance and evaluate recipes contains configurable paramaters for both scripts. These
include:

- Paths for the locations of audio files, metadata and the export location for generated files
- Paramaters for the NAL-R fitting
- Paramaters for the automatic gain control (AGC) compressor used in the baseline enhancer
- Parameters for the challenge evaluator
- Parameters necessary for Hydra to run

The `path.root` parameter defaults to a null value (`???`) and must be overriden with a dataset root path when the
Python script is called in the command line, e.g.

``` bash
user:~$ python mypythonscript.py path.root='/path/to/project'
```

Note the lack of slash at the end of the <code>path.root</code> argument string. If you inspect a variable such as <code>path.metadata_dir</code> you will see that this slash is already included in the line.

``` yaml
path:
  root: ???
  metadata_dir: ${path.root}/clarity_data/metadata

```

The general form for overriding a parameter in the CLI is dot indexed. For the following entry in a <code>config.yaml</code> file:

``` yaml
A:
  B:
    parameter_0: some_value
    parameter_1: some_other_value
```

The CLI syntax to override those values would be:

``` bash
python myscript.py A.B.parameter_0="new_value" A.B.parameter_1="another_new_value"
```

We are now ready to run the prepared Python script `recipes/cec2/baseline/enhance.py` to enhance the audio. However, the
standard configuration is designed to work with the full clarity dataset. We can redirect the script to the correct
folders to use the demo data we have downloaded by overriding the appropriate configuration parameters.

``` bash
python enhance.py \
path.root=${BASE_DIR} \
path.metadata_dir="$\{path.root\}/clarity_data/demo/metadata" \
path.scenes_listeners_file="$\{path.metadata_dir\}/scenes_listeners.demo.json" \
path.listeners_file="$\{path.metadata_dir\}/listeners.json" \
path.scenes_folder="$\{path.root\}/clarity_data/demo/scenes"
```

``` python
from pathlib import Path
import IPython.display as ipd

audio_path = Path("exp/exp/enhanced_signals")
audio_files = list(audio_path.glob("**/*.wav"))

# Listen to a single file
print(audio_files[0])
ipd.Audio(audio_files[0])
```

You can now use the `recipes/cec2/baseline/evaluate.py` script to generate HASPI scores for the signals. The evaluation
is run in the same manner as the enhancement script.

``` bash
python evaluate.py \
path.root=${BASE_DIR} \
path.metadata_dir="$\{path.root\}/clarity_data/demo/metadata" \
path.scenes_listeners_file="$\{path.metadata_dir\}/scenes_listeners.demo.json" \
path.listeners_file="$\{path.metadata_dir\}/listeners.json" \
path.scenes_folder="$\{path.root\}/clarity_data/demo/scenes"
```

Now the HASPI scores have been generated, it is possible to plot the results to assess the improvement imparted by the
signal processing. Start a Python shell (`python` or `ipython`) and paste the following code.

``` python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

unprocessed_si = pd.read_csv("exp/exp/si.csv")

processed_si = pd.read_csv("exp/exp/si_unproc.csv")

data = np.array([processed_si.loc[:, "haspi"], unprocessed_si.loc[:, "haspi"]])
plt.boxplot(np.transpose(data))
plt.title("HASPI Scores")
plt.xticks([1, 2], ["Unprocessed", "Processed"])
plt.show()

```

## 03 Running the CEC2 Baseline from Python

We will be using scene audio and associated metadata. This can be downloaded using the Clarity package's `demo_data` module.

``` python
from clarity.data import demo_data

demo_data.get_metadata_demo()
demo_data.get_scenes_demo()


```

By default, the demo data will have been downloaded into a directory called `clarity_data`.

## Running the baseline

### Importing the baseline NALR and Compressor components

The baseline enhancer is based on [NAL-R prescription fitting](https://pubmed.ncbi.nlm.nih.gov/3743918/). Since output
signals are required to be in 16-bit integer format, a slow acting automatic gain control is implemented to reduce
clipping of the signal introduced by the NAL-R fitting for audiograms which represent more severe hearing loss. The AGC
is followed by a soft-clip function.

The NAL-R and AGC (compressor) classes can be accessed by importing them from the `clarity.enhancer` module.

``` python
from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
```

### Configuring the NALR and Compressor components

To allow for scalable and flexible running on both local and HPC platforms, many Clarity challenge CEC2 scripts and
tools depend on [hydra](https://hydra.cc/) and [submitit](https://github.com/facebookincubator/submitit) for the
configuration of Python code, for the setting of environment variables such as dataset directories, and for enabling
parallelisation of Python on both HPC and local machines. (A full description of how
[hydra](https://hydra.cc/docs/intro/) and [submitit](https://github.com/facebookincubator/submitit) is used in the
Clarity challenges is out of the scope of this tutorial).

In this tutorial, we will be importing the baseline configuration file directly using `omegaconf`. The module can read a
configuration file in YAML format and return a DictConfig object storing the configuration data.

The configuration is included under the `clarity/recipes/cec2/baseline/config.yaml` from when you installed the
`pyclarity** package.

### IMPORTANT

The location of the `recipes` needs a little figuring out and will depend on how you have installed `pyclarity`.

If you installed from PyPI using `pip` under a Miniconda virtual environment called `clarity` configured to
store the virtual environments in the default location of `~/miniconda3/`  then the `recipes` directory will be under
`~/.miniconda/envs/clarity/lib/python-<version>/recipes/`.

If you used a more traditional Virtual Environment which is configured to save environments under `~/.virtualenv` and
called your environment `pyclarity` then the location will be
`~/.virtualenv/pyclarity/lib/python<version>/site-packages/recipes/`.

If you have installed pyclarity from GitHub then the recipes will be under `clarity/recipes/` in the cloned directory.

If you are unsure where to find the files and are on a UNIX like operating system you can search for them with the
following...

``` bash
find ~/ -type f -iname "config.yaml" | grep recipes
```

``` python
from omegaconf import DictConfig, OmegaConf

# IMPORTANT - Modify the following line to reflect where the recipes are located, see notes above
cfg = OmegaConf.load("/home/<username>/.virtualenv/pyclarity/lib/python3.10/site-packages/recipes/cec2/baseline/config.yaml")
assert isinstance(cfg, DictConfig)

```

We will need to override some of the standard paths provided in the baseline `config.yaml` to enable us to run the
baseline on the demo data in this tutorial environment.

We need to supply:

- The root directory of the project data and metadata
- The directory of the metadata
- The directory of the audio data

The default configuration can be overridden by changing the values in the `cfg` object.

``` python
cfg.path["root"] = "clarity_data/demo"
cfg.path["metadata_dir"] = "${path.root}/metadata"
cfg.path["scenes_folder"] = "${path.root}/scenes"
```

(Side note: the Clarity tools come with higher level `recipe` scripts that are designed to be used from the command line. When working with these, default configurations can be overriden by passing command line arguments.)

With the configuration modified, we can now instantiate our `NALR` and `Compressor` objects.

``` python
enhancer = NALR(**cfg.nalr)
compressor = Compressor(**cfg.compressor)

```

### Selecting a scene and a listener

NAL-R fitting involves creating a complementary filterbank that is tuned to the audiogram of a specific listener.

For each scene in the Clarity data set, there are three associated listeners that have been randomly selected, i.e., you are told which listeners to process each scene for. Using the right listeners is particularly important when processing the development (and evaluation) data, i.e., to ensure that your results are comparable with those of others.

The listener audiogram data and the scene-listener associations are defined in the Clarity metadata.

We will first load the scene, targets, listeners and scene_listeners data from the JSON files in which they are stored:

``` python
import json

with open("clarity_data/demo/metadata/scenes.demo.json") as f:
    scene_metadata = json.load(f)

with open("clarity_data/demo/metadata/listeners.json") as f:
    listeners_metadata = json.load(f)

with open("clarity_data/demo/metadata/scenes_listeners.dev.json") as f:
    scene_listeners_metadata = json.load(f)
```

Next, we will select an individual scene from `scenes_metadata`, find its associated listener's and then find the
listener's audiogram data.

So we first choose a scene the the `scene_metadata` list using a `scene_index`, i.e.,

``` python
scene_index = 2
scene = scene_metadata[scene_index]

print(scene)
```

We find the scene's listeners by looking them up in the `scene_listeners_metadata` dict using the scene's `scene_id` as the key.

``` python
scene_id = scene["scene"]
scene_listeners = scene_listeners_metadata[scene_id]

print(scene_listeners)

```

This provides us with the list of `listener_id`s for this scene.

We will select one `listener_id` from this list and use it as the key to select the required listener metadata.

``` python
listener_choice = 1
listener_id = scene_listeners[listener_choice]
listener = listeners_metadata[listener_id]
```

Each listener metadata entry is a dict containing:

- Listener ID
- Audiogram centre frequencies
- Left ear audiogram hearing levels (dBHL)
- Right ear audiogram hearing levels (dBHL)

``` python
print(listener)
```

### Loading the signals to process

Next we will load in the scene audio for the scene that we want to process.

The path to the scenes audio data is stored in the `cfg.path.scenes_folder` variable and the audio files are named with the scene_id as the prefix and using the format.

```text
<SCENE_ID>_<TYPE>_<CHANNEL>.wav
```

where `TYPE` can be `mix`, `target`, `interferer` or `interferer_anechoic`, and `CHANNEL` can be `CH1`, `CH2`, `CH3` or `CH0`.

The baseline system just uses `CH1` (the front microphone of the hearing aid).

Finally, signals are stored as 16-bit integer audio and must be converted to floating point (between -1.0 and 1.0) before use, i.e. by dividing by 2**15.

So, using the `wavfile` module from `scipy.io` to read the file, we have,

``` python
from pathlib import Path

from scipy.io import wavfile

fs, signal = wavfile.read(Path(cfg.path.scenes_folder) / f"{scene_id}_mix_CH1.wav")


signal = signal / 32768.0
```

We can plot the signal to check it looks OK,

``` python
import matplotlib.pylab as plt

plt.plot(signal)
plt.show()
```

### Applying the NALR and Compressor components

We will now build the NALR filterbank according to the audiograms of the listener we have selected and apply the filter to the scene signal. This is done separately for the left and right ear (i.e., for each channel of the stereo scene signal).

``` python
import numpy as np

nalr_fir, _ = enhancer.build(listener["audiogram_levels_l"], listener["audiogram_cfs"])
out_l = enhancer.apply(nalr_fir, signal[:, 0])

nalr_fir, _ = enhancer.build(listener["audiogram_levels_r"], listener["audiogram_cfs"])
out_r = enhancer.apply(nalr_fir, signal[:, 1])

plt.plot(out_l)
plt.show()
```

Following this, slow AGC is applied and a clip detection pass is performed. A tanh function is applied to remove high frequency distortion components from cliipped samples and the files are converted back to 16-bit integer format for saving.

``` python
out_l, _, _ = compressor.process(out_l)
out_r, _, _ = compressor.process(out_r)

enhanced_audio = np.stack([out_l, out_r], axis=1)

plt.plot(enhanced_audio)
plt.show()
```

Finally, the signals are placed through a tanh function which provides a soft-clipping to handle any transient segments that have not been dealt with by the ACG.

The final signals are then converted back into 16-bit format.

``` python
n_clipped = np.sum(np.abs(enhanced_audio) > 1.0)
if n_clipped > 0:
    print(f"{n_clipped} samples clipped")
enhanced_audio = np.tanh(enhanced_audio)
np.clip(enhanced_audio, -1.0, 1.0, out=enhanced_audio)

plt.plot(enhanced_audio)
plt.show()
```

Note, processed signals will be submitted as 16-bit wav-file format, i.e. by first converting back to 16-bit integer format and then saving to file.

```python
signal_16 = (32768.0 * enhanced_audio).astype(np.int16)
```

The standard filename for the processed audio is constructed as

```python
filename = f"{scene['scene']}_{listener['name']}_HA-output.wav"
```

## Evaluating outputs using HASPI

Enhanced scores can now be evaluated using the HASPI speech intelligibility prediction metric and compared to the
unenhanced audio.

HASPI scores are calculated using a 'better ear' approach where left and right signals are acalculated and the higher
score used as the output. The 'better ear' haspi function (`haspi_v2_be`) is imported from `clarity.evaluator.haspi`.

HASPI is an intrusive metric and requires an uncorrupted reference signal. These are provided in the scenes audio data as files with the naming convention `SXXXX_target_CHX.wav`. CH1 is used as the reference transducer for this challenge. We load the file and convert to floating point as before.

``` python
from clarity.evaluator.haspi import haspi_v2_be

fs, reference = wavfile.read(Path(cfg.path.scenes_folder) / f"{scene_id}_target_CH1.wav")

reference = reference / 32768.0
```

We provide the function `haspi_v2_be` with the left and right references, the left and right signals, the sample rate and the audiogram information for the given listener.

Below, we first compute the HASPI score for the unprocessed signal and then for the enhanced signal. We can compute the benefit of the processing by calculating the difference.

``` python
sii_unprocessed = haspi_v2_be(
    xl=reference[:, 0],
    xr=reference[:, 1],
    yl=signal[:, 0],
    yr=signal[:, 1],
    fs_signal=fs,
    audiogram_l=listener["audiogram_levels_l"],
    audiogram_r=listener["audiogram_levels_r"],
    audiogram_cfs=listener["audiogram_cfs"],
)

sii_enhanced = haspi_v2_be(
    xl=reference[:, 0],
    xr=reference[:, 1],
    yl=enhanced_audio[:, 0],
    yr=enhanced_audio[:, 1],
    fs_signal=fs,
    audiogram_l=listener["audiogram_levels_l"],
    audiogram_r=listener["audiogram_levels_r"],
    audiogram_cfs=listener["audiogram_cfs"],
)

print(f"Original audio HASPI score is {sii_unprocessed}")

print(f"Enhanced audio HASPI score is {sii_enhanced}")

print(f"Improvement from processing is {sii_enhanced-sii_unprocessed}")
```

For the scene and listener we have selected the original HASPI score should be about `0.081` and the score after
enhancement should be about `0.231`. Note, HASPI uses internal masking noise and because we have not set the random
seed, scores may vary a little from run to run - the variation should not be more than `+-0.0005` and often much less.

Note also that the 'enhanced' score is still very low - this is not surprising given that the processing is only
amplying amplification and compression. There is no noise cancellation, no multichannel processing, etc, etc. The
purpose of the enhancement challenge is to add these components in order to try and improve on this baseline.

Good luck!
