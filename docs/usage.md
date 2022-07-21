# Usage

Before getting started you should ensure you have installed pyClarity within a virtual environment, if you've not
already done so please refer to the [installation instructions](installation.md).

To check run `pip show pyclarity` and you should get some information about the installed version. If you don't check that
you have activated the Virtual Environment you made the install under, if the package still isn't found then you should
go through the installation process again within your Virtual Environment.

## Tutorials

If you prefer using Jupyter Notebooks you can run the [pyClarity Tutorials](https://claritychallenge.org/tutorials) in
Google CoLab. These demonstrate two ways of interacting with the tools provided.


## Command Line Interface

The examples and code below take you through the [tutorials]() and use smaller demo datasets which are provided under
`clarity.data.demo_data` and have specific functions for loading.

### Installing pyClarity and Using Metadata

This demonstration uses only the `metadata` datasets and it is downloaded to the `clarity_data/demo/metadata/` directory.

``` python
from clarity.data import demo_data

demo_data.get_metadata_demo()
```

This will have created a directory called `clarity_data` containing the metadata files that have been downloaded.

#### The structure of the metadata files

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
print(f'Value of SNR for scene_0 : {scene_0["SNR"]})
# Directly...
print(f'Value of SNR for scene_0 : {scenes[0]["SNR"]})
```

#### Processing Collections of Scenes

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

#### Example: Locating information about the scene's listener

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
```
