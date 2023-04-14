"""Code for building the scenes.json files."""
from __future__ import annotations

import itertools
import json
import logging
import math
import random
import re
from enum import Enum
from typing import Final

import numpy as np
from tqdm import tqdm

# A logger for this file
log = logging.getLogger(__name__)


# Get json output to round to 4 dp
json.encoder.c_make_encoder = None  # type: ignore


class RoundingFloat(float):
    """Round a float to 4 decimal places."""

    __repr__ = staticmethod(lambda x: format(x, ".4f"))  # type: ignore


json.encoder.float = RoundingFloat  # type: ignore

# rpf file Handling

N_SCENES: Final = 10000  # Number of scenes to expect
N_INTERFERERS: Final = 3  # Default number of interferers to expect


def set_random_seed(random_seed):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)


def get_vector(text: str, vector_name: str) -> list[float]:
    """Get a vector quantity from the rpf file.
    Will read rpf vector quantities, eg.
    "sourceViewVectors = -0.095,-0.995, 0.000"

    Args:
        text (str): string contents of the rpf file
        vector_name (str): name of vector to extract (e.g. "sourceViewVectors")

    Returns:
        List[float]: vector as list of floats
    """
    findall_str = f".*{vector_name}.*"
    x_str = re.findall(findall_str, text)
    x_str = re.sub(" ", "", x_str[0].split("=")[-1])  # type: ignore
    x = [float(idx) for idx in x_str.split(",")]  # type: ignore
    return x


def get_room_dims(text: str) -> list:
    """Find the room dimensions in the rpf file.

    Args:
        text (str): String to be searched for room dimensions (string to be searched
            for is of the form 'ProjectName = CuboidRoom_5.9x3.4186x2.9').


    Returns:
        list: List of the three dimensions of the room.
    """
    # Room dimensions appear in the file like this,
    #    ProjectName = CuboidRoom_5.9x3.4186x2.9
    roomdims = re.findall(r"ProjectName = .*", text)[0]
    return roomdims.split("=")[-1].split("_")[-1]


def get_room_name(text: str) -> str:
    """Find the room name in the rpf file.

    Args:
        text (str): String to be searched for room name ('R' followed by 5 digits).

    Returns:
        str: The room name.
    """
    return re.findall(r"R\d\d\d\d\d", text)[0]


def read_rpf_file(rpf_filename: str) -> dict:
    """Process an rpf file and return key contents as a dictionary.

    Args:
        rpf_filename (str): Path to an rpf file to be read.

    Returns:
        dict: dictionary of rpf file contents
            {"position": sourcePositions, "view_vector": sourceViewVectors}
    """
    with open(rpf_filename, encoding="utf-8") as f:
        text = f.read()

    rpf_dict = {}

    room_dict = {}
    room_dict["name"] = get_room_name(text)  # type: ignore
    room_dict["dimensions"] = get_room_dims(text)  # type: ignore

    rpf_dict["room"] = room_dict

    # This hrir is not needed - added later
    # rpf_dict["hrirfilename"] = get_hrir_filename(text)

    rpf_dict["source"] = {
        "position": get_vector(text, "sourcePositions"),  # type: ignore
        "view_vector": get_vector(text, "sourceViewVectors"),  # type: ignore
    }

    rpf_dict["receiver"] = {
        "position": get_vector(text, "receiverPositions"),  # type: ignore
        "view_vector": get_vector(text, "receiverViewVectors"),  # type: ignore
    }

    return rpf_dict


def build_room(target_file: str, interferer_files: list[str]) -> dict:
    """Build room json file from contents of related rpf files.
    Note, there is an rpf file for each source in the scene. All of these
    files are read and a single scene json file is constructed.

    Args:
        target_file (str): rpf file containing the target position
        interferer_files (List[str]): list of files containing the interferer positions

    Returns:
        dict: dictionary representation of the scene following CEC2 scene.json format
    """
    rpf_dict = read_rpf_file(target_file)

    scene_dict: dict = {}
    scene_dict["name"] = rpf_dict["room"]["name"]
    scene_dict["dimensions"] = rpf_dict["room"]["dimensions"]
    scene_dict["target"] = rpf_dict["source"]
    scene_dict["listener"] = rpf_dict["receiver"]

    # Read interferer rpf files and extract source info
    interferer_rpfs = [
        read_rpf_file(interferer_file) for interferer_file in interferer_files
    ]
    interferers = [interferer_rpf["source"] for interferer_rpf in interferer_rpfs]
    # The source view vectors are removed as interferer are all omnidirectional
    for interferer in interferers:
        del interferer["view_vector"]
    scene_dict["interferers"] = interferers

    return scene_dict


def make_rpf_filename_dict(
    rpf_location: str, scene_str: str, n_interferers: int
) -> dict:
    """Construct dictionary storing all rpf files that will be processed.

    Args:
        rpf_location (str): Location
        scene (str): Scene (as string)
        n_interferers (int): Number of interferers.

    Returns:
        dict: Dictionary of rpf files to be processed.
    """
    rpf_filename_format = "{rpf_location}/{scene_str}_{source}.rpf"

    rpf_filedict = {}
    rpf_filedict["target"] = rpf_filename_format.format(
        rpf_location=rpf_location, scene_str=scene_str, source="t"
    )

    rpf_filedict["interferers"] = [
        rpf_filename_format.format(
            rpf_location=rpf_location, scene_str=scene_str, source=f"i{i}"
        )
        for i in range(1, n_interferers + 1)
    ]  # type: ignore
    return rpf_filedict


# target handling
def get_num_pre_samples(pre_samples_range: list) -> int:
    """Number of samples prior to target onset.

    Args:
        pre_samples_range (list): parameters for number of samples prior to target onset

    Returns:
    """
    return random.randint(*pre_samples_range)


def get_num_post_samples(post_samples_range: list) -> int:
    """Number of samples to continue player after target offsets.


    Args:
        post_samples_range (list): parameters for number of samples to continue.

    Returns:
    """
    return random.randint(*post_samples_range)


def add_this_target_to_scene(
    target: dict, scene: dict, pre_samples_range: list, post_samples_range: list
):
    """Add the target details to the scene dict.

    Adds given target to given scene. Target details will be taken from the target dict
    but the start time will be according to the CEC2 target start time specification.

    Args:
        target (dict): target dict read from target metadata file.
        scene (dict): complete scene dictionary.
        pre_samples_range (list): parameters for number of samples prior to target onset
        post_samples_range (list): parameters for number of samples to continue
            player after target offsets.
    """
    scene_target = {}
    num_pre_samples = get_num_pre_samples(pre_samples_range)
    num_post_samples = get_num_post_samples(post_samples_range)
    scene_target["name"] = target["wavfile"]
    scene_target["time_start"] = num_pre_samples
    scene_target["time_end"] = num_pre_samples + target["nsamples"]
    scene["target"] = scene_target
    scene["duration"] = num_pre_samples + target["nsamples"] + num_post_samples


# SNR handling
def generate_snr(snr_range: list[int]) -> float:
    """Generate a random Signal Noise Ratio (SNR).

    Args:
        snr_range (list): Range from which to uniformly sample SNR.

    Returns:
        float: random number from uniform distribution in given range.
    """
    return random.uniform(*snr_range)


# Interferer handling
class InterfererType(Enum):
    """Enum for interferer types."""

    SPEECH = "speech"
    NOISE = "noise"
    MUSIC = "music"


def select_interferer_types(allowed_n_interferers: list) -> list[InterfererType]:
    """Select the interferer types to use.

    The number of interferer is drawn randomly from list of allowed valued.
    The type of each is chosen randomly but there is not allowed to be
    more than 1 music source.

    Args:
        allowed_n_interferers (list): list of allowed number of interferers

    Returns:
        list(InterfererType): list of interferer types to use
    """

    def valid(selection):
        """Check if the selection is valid."""
        return selection.count(InterfererType.MUSIC) <= 1

    n_interferers = random.choice(allowed_n_interferers)
    selection = None
    while selection is None or not valid(selection):
        selection = random.choices(list(InterfererType), k=n_interferers)
    return selection


def select_random_interferer(
    interferers: list[list], dataset: str, required_samples: int
) -> dict:
    """Randomly select an interferer.
    Interferers stored as list of list. First randomly select a sublist
    then randomly select an item from sublist matching constraints.

    Args:
        interferers (list(list)): interferers as list of lists
        dataset (str): desired data [train, dev, eval]
        required_samples (int): required number of samples

    Raises:
        ValueError: if no suitable interferer is found

    Returns:
        dict: the interferer dict
    """
    interferer_group = random.choice(interferers)
    filtered_interferer_group = [
        i
        for i in interferer_group
        if i["dataset"] == dataset and i["nsamples"] >= required_samples
    ]
    try:
        interferer = random.choice(filtered_interferer_group)
    except IndexError as e:
        raise ValueError(
            f"No suitable interferer found for dataset {dataset} and required samples"
            f" {required_samples}"
        ) from e
    return interferer


def get_random_interferer_offset(interferer: dict, required_samples: int) -> int:
    """Generate a random offset sample for interferer.
    The offset sample is the point within the masker signal at which the interferer
    segment will be extracted. Randomly selected but with care for it not to start
    too late, i.e. such that the required samples would overrun the end of the masker
    signal will be used is taken.

    Args:
        interferer (dict): the interferer metadata
        required_samples (int): number of samples that is going to be required

    Returns:
        int: a valid randomly selected offset
    """
    masker_nsamples = interferer["nsamples"]
    latest_start = masker_nsamples - required_samples
    if latest_start < 0:
        log.error("Interferer %s does not has enough samples.", interferer["ID"])

    assert (
        latest_start >= 0
    )  # This should never happen - mean masker was too short for the scene
    return random.randint(0, latest_start)


def add_interferer_to_scene_inner(
    scene: dict,
    interferers: dict,
    number: list,
    start_time_range: list[int],
    end_early_time_range: list[int],
):
    """Randomly select interferers and add them to the given scene.
    A random number of interferers is chosen, then each is given a random type
    selected from the possible speech, nonspeech, music types.
    Interferers are then chosen from the available lists according to the type
    and also taking care to match the scenes 'dataset' field, ie. train, dev, test.
    The interferer data is supplied as a dictionary of lists of lists. The key
    being "speech", "nonspeech", or "music", and the list of list being a partitioned
    list of interferers for that type.
    The idea of using a list of lists is that interferers can be split by
    subcondition and then the randomization draws equally from each subcondition,
    e.g. for nonspeech there is "washing machine", "microwave" etc. This ensures that
    each subcondition is equally represented even if the number of exemplars of
    each subcondition is different.
    Note, there is no return. The scene is modified in place.

    Args:
        scene (dict): the scene description
        interferers (dict): the interferer metadata
        number (list): number of interferers
        start_time_range (list): range of starting points as integers, a random number
            is selected between these.
        end_early_time_range (list): range of end points as integers, a random number
            is selected between these.
    """
    dataset = scene["dataset"]
    selected_interferer_types = select_interferer_types(number)
    n_interferers = len(selected_interferer_types)
    n_positions_available = len(scene["room"]["interferers"])
    available_positions = range(1, n_positions_available + 1)
    positions = random.sample(available_positions, n_interferers)
    positions.sort()

    # Make list of empty dicts match the number selected
    scene["interferers"] = [{"position": position} for position in positions]

    # Randomly instantiate each interferer in the scene
    for scene_interferer, scene_type in zip(
        scene["interferers"], selected_interferer_types
    ):
        desired_start_time = random.randint(*start_time_range)

        scene_interferer["time_start"] = min(scene["duration"], desired_start_time)
        desired_end_time = scene["duration"] - random.randint(*end_early_time_range)

        scene_interferer["time_end"] = max(
            scene_interferer["time_start"], desired_end_time
        )

        required_samples = scene_interferer["time_end"] - scene_interferer["time_start"]
        interferer = select_random_interferer(
            interferers[scene_type], dataset, required_samples
        )
        scene_interferer["type"] = scene_type.value
        scene_interferer["name"] = interferer["ID"]
        scene_interferer["offset"] = get_random_interferer_offset(
            interferer, required_samples
        )


# listener handlinf


def get_random_hrir_set(heads, channels):
    """Get a random HRIR set."""
    head = random.choice(heads)
    hrir_names = [f"{head}-{channel}" for channel in channels]
    return hrir_names


def generate_rotation(
    scene: dict,
    relative_start_time_range: list,
    duration_mean: float,
    duration_sd: float,
    angle_initial_mean: float,
    angle_initial_sd: float,
    angle_final_range: tuple,
) -> list[dict]:
    """Generate a suitable head rotation for the given scene.
    Based on behavioural studies by Hadley et al. TODO: find ref

    Args:
        scene (dict): the scene description
        relative_start_time_range (list): Range from which start time is
            uniformly selected at random.
        duration_mean (float): mean of the time offset for start of turn
        duration_sd (float): standard deviation of the time offset for start of turn
        angle_initial_mean (float):
        angle_initial_sd (float):
        angle_final_range (tuple):

    Returns:
        list(dict): list of dicts with keys "sample" and "view_vector"
        specifying the head motion.
    """

    # Generate the random start and end time for the head rotation
    start_offset = random.uniform(*relative_start_time_range)
    duration = int(random.gauss(duration_mean, duration_sd))
    start_time = scene["target"]["time_start"] + start_offset
    start_time = max(start_time, 0)
    end_time = start_time + duration

    # Head initially offset to left or right at random
    random_sign = 1 if random.random() < 0.5 else -1

    # Generate the random start and end angle offsets for the head rotation
    offset = 0.0
    while True:
        offset = random.gauss(0, angle_initial_sd)
        # All rotations kept within 2 sigma of the mean
        if abs(offset) < 2 * angle_initial_sd:
            break

    angle_initial_offset = random_sign * math.radians(angle_initial_mean + offset)
    angle_final_offset = random_sign * (
        math.radians(random.uniform(*angle_final_range))
    )

    # Convert angle from relative to target speaker to room axes
    listener_pos = scene["room"]["listener"]["position"]
    target_pos = scene["room"]["target"]["position"]
    delta_x = target_pos[0] - listener_pos[0]
    delta_y = target_pos[1] - listener_pos[1]
    angle_target = math.atan2(delta_y, delta_x)

    # NOTE: JSON file will store angles in degrees
    angle_final = math.degrees(angle_target + angle_final_offset)
    angle_initial = math.degrees(angle_target + angle_initial_offset)

    # # Convert angles back to view vectors for the return object
    # start_view_vector = [math.cos(angle_initial), math.sin(angle_initial), 0.0]
    # end_view_vector = [math.cos(angle_final), math.sin(angle_final), 0.0]
    rotation = [
        {"sample": start_time, "angle": angle_initial},
        {"sample": end_time, "angle": angle_final},
    ]
    return rotation


class RoomBuilder:
    """Functions for handling rooms."""

    def __init__(self):
        """Initialize the room builder."""
        self.rooms = []
        self.room_dict = {}

    def get_room(self, name: str):
        """Get a room by name.

        Args:
            name (str): Name of room to extract.

        Returns:
        """
        return self.room_dict[name]

    def rebuild_dict(self):
        """Build room dictionary."""
        self.room_dict = {room["name"]: room for room in self.rooms}

    def build_from_rpf(
        self, rpf_location, n_interferers=N_INTERFERERS, n_rooms=N_SCENES, start_room=1
    ):
        """Build a list of rooms by extracting info from RAVEN rpf files.

        Args:
            rpf_location (str): path to where rpf files are stored
            n_interferers (int, optional): number of interferer definitions to expect.
              Defaults to N_INTERFERERS.
            n_rooms (int, optional): number of scenes to expect. Defaults to N_SCENES.
            start_room (int, optional): index of the first room to expect
        """
        # Construct all rpf filenames for given number of interferers
        rpf_filedicts = [
            make_rpf_filename_dict(rpf_location, f"R{n:05d}", n_interferers)
            for n in range(start_room, start_room + n_rooms)
        ]

        # Process all rpf file to generate the list of rooms dicts
        self.rooms = [
            build_room(rpf_filedict["target"], rpf_filedict["interferers"])
            for rpf_filedict in tqdm(rpf_filedicts)
        ]
        self.rebuild_dict()

    def load(self, filename) -> None:
        """Load the list of room from a JSON file.

        Args:
            filename (str): filename to JSON file from which room data is loaded.
        """
        with open(filename, encoding="utf-8") as f:
            self.rooms = json.load(f)
        self.rebuild_dict()

    def save_rooms(self, filename):
        """Save the list of rooms to a JSON file.

        Args:
            filename (str): filename to JSON file to which room data is saved.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.rooms, f, indent=2)


class SceneBuilder:
    """Class with methods for building a list of scenes."""

    def __init__(
        self,
        rb,
        scene_datasets,
        target,
        interferer,
        snr_range,
        listener,
        shuffle_rooms=None,
    ):
        """Initialise SceneBuilder class.

        Args:
            rb ():
            scene_datasets (dict): Dictionary of parametesr for initialising scenes.
            target (dict): Dictionary of parameters for adding target to scenes.
            interferer (dict): Dictionary of parameters for adding interferer to scenes.
            snr_range (list): Range of values from which to sample Signal Noise Ratio.
            listener (dict): Dictionary of parameters for adding listener to scenes.
            shuffle_rooms=None ():
        """
        self.rb = rb
        self.scenes = []
        self.scene_datasets = scene_datasets
        self.target = target
        self.interferer = interferer
        self.snr_range = snr_range
        self.listener = listener
        self.shuffle_rooms = shuffle_rooms

    def save_scenes(self, filename: str) -> None:
        """Save the list of scenes to a JSON file.

        Args:
            filename (str): Filename to save scenes to.

        Returns:
            None
        """
        scenes = self.scenes.copy()
        # Replace the room structure with the room ID
        for scene in scenes:
            scene["room"] = scene["room"]["name"]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.scenes, f, indent=2)

    def instantiate_scenes(self, dataset) -> None:
        """Instantiate scenes with targets, interferers, SNR and listeners.

        Args:
            dataset:
        """
        log.info(f"Initialise {dataset} scenes")
        self.initialise_scenes(dataset, **self.scene_datasets)
        log.info("Adding targets to scenes")
        self.add_target_to_scene(dataset, **self.target)
        log.info("Adding interferers to scenes")
        self.add_interferer_to_scene(**self.interferer)
        log.info("Assigning an SNR to each scene")
        self.add_SNR_to_scene(self.snr_range)
        log.info("Adding listener behaviours to scenes")
        self.add_listener_details_to_scene(**self.listener)

    def initialise_scenes(
        self, dataset, n_scenes: int, room_selection: str, scene_start_index: int
    ):
        """
        Initialise the scenes for a given dataset.

        Args:
            dataset: train, dev, or eval set
            n_scenes (int): number of scenes to generate
            room_selection (str): SEQUENTIAL or RANDOM
            scene_start_index (int): index to start for scene IDs

        Raises: TypeError if room_selection is not SEQUENTIAL or RANDOM
        """
        rooms = self.rb.rooms.copy()
        if self.shuffle_rooms:
            random.shuffle(rooms)

        # Construct the scenes adding the room and dataset label
        self.scenes = []
        scenes = [{"dataset": dataset} for _ in range(n_scenes)]
        # Sequential mode: cycle through all available rooms
        if room_selection == "SEQUENTIAL":
            # Cycle through rooms if not enough for required number of scenes
            for scene, room in zip(scenes, itertools.cycle(rooms)):
                scene["room"] = room
        # Random mode: randomly select rooms without replacement
        elif room_selection == "RANDOM":
            for scene in scenes:
                scene["room"] = random.choice(rooms)
        else:
            raise TypeError(f"Invalid room selection mode: {room_selection}")
        self.scenes.extend(scenes)

        # Set the scene ID
        for index, scene in enumerate(self.scenes, scene_start_index):
            scene["scene"] = f"S{index:05d}"

    def add_target_to_scene(
        self,
        dataset: str,
        target_speakers: str,
        target_selection: str,
        pre_samples_range: list,
        post_samples_range: list,
    ):
        """Add target info to the scenes.

        Uses target speaker file set via config.

        Args:
            dataset (str): dataset to be added.
            target_speakers (str):
            target_selection (str): Type of target to be added, valid values are
                'SEQUENTIAL' and 'RANDOM'.
            pre_samples_range (list): Parameters for number of samples prior to target
                onset.
            post_samples_range (list): Parameters for number of samples to continue
                player after target offsets.

        Raises: TypeError if room_selection is not SEQUENTIAL or RANDOM
        """
        with open(target_speakers, encoding="utf-8") as f:
            targets = json.load(f)

        targets_dataset = [t for t in targets if t["dataset"] == dataset]
        scenes_dataset = [s for s in self.scenes if s["dataset"] == dataset]

        random.shuffle(targets_dataset)

        if target_selection == "SEQUENTIAL":
            # Sequential mode: Cycle through targets sequentially
            for scene, target in zip(scenes_dataset, itertools.cycle(targets_dataset)):
                add_this_target_to_scene(
                    target, scene, pre_samples_range, post_samples_range
                )
        elif target_selection == "RANDOM":
            # Random mode: randomly select target with replacement
            for scene in scenes_dataset:
                add_this_target_to_scene(
                    random.choice(targets_dataset),
                    scene,
                    pre_samples_range,
                    post_samples_range,
                )
        else:
            raise TypeError(f"Invalid room selection mode: {target_selection}")

    def add_SNR_to_scene(self, snr_range: list):
        """Add the Signal Noise Ratio (SNR) info to the scenes.

        Args:
            snr_range (list): Range of values from which SNR will be sampled.

        Returns:
        """
        for scene in tqdm(self.scenes):
            scene["SNR"] = generate_snr(snr_range)

    def add_interferer_to_scene(
        self,
        speech_interferers: str,
        noise_interferers: str,
        music_interferers: str,
        number: list,
        start_time_range: list,
        end_early_time_range: list,
    ):
        """Add interferer to the scene description file.

        Args:
            speech_interferers (str): Path to speech interferer to load.
            noise_interferers (str): Path to noise interferer to load.
            music_interferers (str): Path to music interferer to load
            number (list): Number of interefers to be added.
            start_time_range (list): Range for randomly selecting start point.
            end_early_time_range (list): Range for randomly selecting end point.

        Returns:
            None
        """
        # Load and prepare speech interferer metadata
        with open(speech_interferers, encoding="utf-8") as f:
            interferers_speech = json.load(f)
        for interferer in interferers_speech:
            interferer["ID"] = (
                interferer["speaker"] + ".wav"
            )  # selection require a unique "ID" field
        # Selection process requires list of lists
        interferers_speech = [interferers_speech]

        # Load and prepare noise (i.e. noise) interferer metadata
        with open(noise_interferers, encoding="utf-8") as f:
            interferers_noise = json.load(f)
        for interferer in interferers_noise:
            interferer["ID"] += ".wav"
        interferer_by_type: dict = {}
        for interferer in interferers_noise:
            interferer_by_type.setdefault(interferer["class"], []).append(interferer)
        interferers_noise = list(interferer_by_type.values())

        # Load and prepare music interferer metadata
        with open(music_interferers, encoding="utf-8") as f:
            interferers_music = json.load(f)
        for interferer in interferers_music:
            interferer["ID"] = interferer[
                "file"
            ]  # selection require a unique "ID" field
        interferers_music = [interferers_music]

        interferers = {
            InterfererType.SPEECH: interferers_speech,
            InterfererType.NOISE: interferers_noise,
            InterfererType.MUSIC: interferers_music,
        }

        for scene in tqdm(self.scenes):
            add_interferer_to_scene_inner(
                scene, interferers, number, start_time_range, end_early_time_range
            )

    def add_listener_details_to_scene(
        self,
        heads,
        channels,
        relative_start_time_range: list,
        duration_mean: float,
        duration_sd: float,
        angle_initial_mean: float,
        angle_initial_sd: float,
        angle_final_range: tuple,
    ):
        """Add the listener info to the scenes.

        Args:
            heads ():
            channels ():
            relative_start_time_range (list): Range from which start time is selected at
                random.
            duration_mean (float): mean of the time offset for start of turn
            duration_sd (float): standard deviation of the time offset for start of turn
            angle_initial_mean (float):
            angle_initial_sd (float):
            angle_final_range (tuple):

        Returns:
            None
        """
        for scene in tqdm(self.scenes):
            listener = {}
            listener["rotation"] = generate_rotation(
                scene,
                relative_start_time_range,
                duration_mean,
                duration_sd,
                angle_initial_mean,
                angle_initial_sd,
                angle_final_range,
            )
            listener["hrir_filename"] = get_random_hrir_set(heads, channels)
            scene["listener"] = listener
