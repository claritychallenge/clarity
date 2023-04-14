"""
Class to generate random parameters for the Car noise signal generation

These are 2 separated class to keep the logic separated
"""
from __future__ import annotations

from typing import Final

import numpy as np


class CarNoiseParametersGenerator:
    """
    A class to generate noise parameters for a car.
    The constructor takes a boolean flag to indicate whether some
    parameters should be randomized or not.

    The method `gen_parameters` takes a speed in kilometers per hour
    and returns a dictionary of noise parameters.

    The global variables `GEAR_LOOKUP` and `RPM_LOOKUP`
    are used to determine the gear and RPM for a given speed.

    Example:
        >>> car_noise_parameters = CarNoiseParameters(random_flag=True)
        >>> parameters = car_noise_parameters.gen_parameters(speed_kph=100)
        >>> parameters
        {
        "speed": 70,
        "gear": 4,
        "rpm": 1890.0,
        "primary_filter": {
            "order": 1,
            "btype": "lowpass",
            "cutoff_hz": 12.685320000000003,
        },
        "secondary_filter": {
            "order": 2,
            "btype": "lowpass",
            "cutoff_hz": 276.22400000000005,
        },
        "bump": {"order": 1, "btype": "bandpass", "cutoff_hz": [34, 66.64000000000001]},
        "dip_low": {"order": 1, "btype": "lowpass", "cutoff_hz": 90},
        "dip_high": {"order": 1, "btype": "highpass", "cutoff_hz": 220.12494775682322},
    }
    """

    GEAR_LOOKUP: Final = {110: [6], 100: [5, 6], 85: [5], 75: [4, 5], 60: [4], 50: [3]}
    RPM_LOOKUP: Final = {6: 0.28, 5: 0.34, 4: 0.45, 3: 0.60}
    REFERENCE_CONSTANT_DB: Final = 30

    def __init__(
        self,
        primary_filter_cutoff: float = 0.14,
        primary_filter_constant_hz: float = 2.86,
        secondary_filter_cutoff: float = 0.8,
        secondary_filter_constant_hz: float = 200,
        random_flag: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """
        Constructor takes a boolean flag to indicate whether
        some parameters should be randomized or not.

        It preset several speed independent parameters for the noise generation.

        Args:
            primary_filter_cutoff (float, optional): The speed dependent cutoff in Hz
                for Primary Filter
            primary_filter_constant_hz (float, optional): The constant cutoff in Hz
                for Primary Filter
            secondary_filter_cutoff (float, optional): The speed dependent cutoff in Hz
                for Secondary Filter
            secondary_filter_constant_hz (float, optional): The constant cutoff in Hz
                for Secondary Filter
            random_flag (bool, optional): Flag to indicate whether some parameters
                should be randomized or not
            random_seed (int, optional): Random seed to use for randomization

        """

        self.random_flag = random_flag

        if self.random_flag and random_seed is not None:
            np.random.seed(random_seed)

        # .. randomization range for frequency multiplier
        self.randomisation_range_freq_multiplier = (
            np.arange(0.9, 1.1, 0.001) if self.random_flag else np.array([1.0])
        )

        # primray filter .. fix at 6 dB per octave and a
        # slight speed-dependent Hz + randomization
        self.primary_filter = {
            "speeddependence_cutoff_hzperkph": primary_filter_cutoff,
            "constant_hz": primary_filter_constant_hz,
        }

        # .. secondary filter .. fix at 12 dB per octave and a
        # slight speed-dependent Hz + randomization
        self.secondary_filter = {
            "speeddependence_cutoff_hzperkph": secondary_filter_cutoff,
            "constant_hz": secondary_filter_constant_hz,
        }

        # bump
        self.bump_lower_limit_for_randomization_db = 0 if self.random_flag else 6
        self.bump_upper_limit_for_randomization_db = 6

    def set_new_randomisation_range_freq_multiplier(self):
        """Set a new frequency multiplier use in the
        primary and secondary filters
        """
        self.randomisation_range_freq_multiplier = (
            np.arange(0.9, 1.1, 0.001) if self.random_flag else np.array([1.0])
        )

    def gen_parameters(self, speed_kph: float) -> dict:
        """
        Generate a dictionary of noise parameters for a given speed.

        Parameters
        ----------
        speed_kph : float
            The speed of the car in kilometers per hour between 50 and 120 km/h.

        Returns
        -------
        parameters : dict
            A dictionary of noise parameters.
        """

        if not 50 <= speed_kph <= 120:
            raise ValueError("Speed should be between 50 and 120 kph.")

        gear = self._get_gear(speed_kph)
        rpm = self._get_rpm(gear, speed_kph)

        reference_level_db = self.REFERENCE_CONSTANT_DB
        if self.random_flag:
            reference_level_db += np.random.choice(np.arange(0, 3.1, 0.1))

        engine_num_harmonics = (
            25 if not self.random_flag else np.random.choice(np.arange(10, 41))
        )

        primary_filter = self._generate_primary_filter(speed_kph)
        secondary_filter = self._generate_secondary_filter(speed_kph)
        bump_filter = self._generate_bump_filter()
        dip_filter_low, dip_filter_high = self._generate_dip_filter()

        return {
            "speed": float(speed_kph),
            "gear": int(gear),
            "reference_level_db": float(reference_level_db),
            "engine_num_harmonics": int(engine_num_harmonics),
            "rpm": float(rpm),
            "primary_filter": primary_filter,
            "secondary_filter": secondary_filter,
            "bump": bump_filter,
            "dip_low": dip_filter_low,
            "dip_high": dip_filter_high,
        }

    def _get_gear(self, speed_kph: float) -> int:
        for speed, possible_gears in self.GEAR_LOOKUP.items():
            if speed_kph >= speed:
                return int(
                    possible_gears[-1]
                    if not self.random_flag
                    else np.random.choice(possible_gears)
                )
        return 0

    def _get_rpm(self, gear: int, speed_kph: float) -> float:
        rpm = self.RPM_LOOKUP[gear] * speed_kph * 60
        return rpm

    def _generate_primary_filter(self, speed_kph: float) -> dict:
        return {
            "order": 1,
            "btype": "lowpass",
            "cutoff_hz": (
                self.primary_filter["speeddependence_cutoff_hzperkph"] * speed_kph
                + self.primary_filter["constant_hz"]
            )
            * np.random.choice(self.randomisation_range_freq_multiplier),
        }

    def _generate_secondary_filter(self, speed_kph: float) -> dict:
        return {
            "order": 2,
            "btype": "lowpass",
            "cutoff_hz": (
                self.secondary_filter["speeddependence_cutoff_hzperkph"] * speed_kph
                + self.secondary_filter["constant_hz"]
            )
            * np.random.choice(self.randomisation_range_freq_multiplier),
        }

    def _generate_bump_filter(self) -> dict:
        bump_filter_order = 1 if not self.random_flag else np.random.choice([1, 2])
        bump_filter_f1_hz = (
            30 if not self.random_flag else np.random.randint(20, 101, size=1)[0]
        )
        bump_filter_f2_hz = (
            60
            if not self.random_flag
            else bump_filter_f1_hz * np.random.choice(np.arange(1.25, 2.01, 0.01))
        )
        filter_dict = {
            "order": int(bump_filter_order),
            "btype": "bandpass",
            "cutoff_hz": [int(bump_filter_f1_hz), int(bump_filter_f2_hz)],
        }
        return filter_dict

    def _generate_dip_filter(self) -> tuple[dict, dict]:
        dip_filter_order = 2 if not self.random_flag else np.random.choice([1, 2])
        dip_filter_f1_hz = (
            200 if not self.random_flag else np.random.choice(range(30, 201, 10))
        )
        dip_filter_f2_hz = (
            300 if not self.random_flag else dip_filter_f1_hz * np.random.uniform(2, 3)
        )
        filter_dict_low = {
            "order": int(dip_filter_order),
            "btype": "lowpass",
            "cutoff_hz": int(dip_filter_f1_hz),
        }
        filter_dict_high = {
            "order": int(dip_filter_order),
            "btype": "highpass",
            "cutoff_hz": int(dip_filter_f2_hz),
        }

        return filter_dict_low, filter_dict_high
