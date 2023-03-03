"""
Class to generate random parameters for the Car noise signal generation

These are 2 separated class to keep the logic separated
"""
from typing import Any, Dict, Tuple

import numpy as np


class CarNoiseParameters:
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

    GEAR_LOOKUP = {110: [6], 100: [5, 6], 85: [5], 75: [4, 5], 60: [4], 50: [3]}
    RPM_LOOKUP = {6: 0.28, 5: 0.34, 4: 0.45, 3: 0.60}

    def __init__(self, random_flag=True):
        """
        Constructor takes a boolean flag to indicate whether
        some parameters should be randomized or not.

        It preset several speed independent parameters for the noise generation.

        """

        self.random_flag = random_flag

        # .. randomization range for frequency multiplier
        self.randomisationrange_freqmultiplier = (
            np.arange(0.9, 1.1, 0.001) if self.random_flag else [1]
        )

        # primray filter .. fix at 6 dB per octave and a
        # slight speed-dependent Hz + randomization
        self.primary_filter = {
            "speeddependence_cutoff_hzperkph": 0.14,
            "constant_hz": 2.86,
        }

        # .. secondary filter .. fix at 12 dB per octave and a
        # slight speed-dependent Hz + randomization
        self.secondary_filter = {
            "speeddependence_cutoff_hzperkph": 0.8,
            "constant_hz": 200,
        }

        # bump
        self.bump_lowerlimitforrandomization_db = 0 if self.random_flag else 6
        self.bump_upperlimitforrandomization_db = 6

    def set_new_randomisationrange_freqmultiplier(self):
        """Set a new freqeuncy multiplier use in the
        primary and secondary filters
        """
        self.randomisationrange_freqmultiplier = (
            np.arange(0.9, 1.1, 0.001) if self.random_flag else [1]
        )

    def gen_parameters(self, speed_kph: float) -> Dict:
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

        primary_filter = self._generate_primary_filter(speed_kph)
        secondary_filter = self._generate_secondary_filter(speed_kph)
        bump_filter = self._generate_bump_filter()
        dip_filter_low, dip_filter_high = self._generate_dip_filter()

        parameters = {
            "speed": float(speed_kph),
            "gear": int(gear),
            "rpm": float(rpm),
            "primary_filter": primary_filter,
            "secondary_filter": secondary_filter,
            "bump": bump_filter,
            "dip_low": dip_filter_low,
            "dip_high": dip_filter_high,
        }

        return parameters

    def _get_gear(self, speed_kph: float) -> Any[int, None]:
        for speed, possible_gears in self.GEAR_LOOKUP.items():
            if speed_kph >= speed:
                return (
                    possible_gears[-1]
                    if not self.random_flag
                    else np.random.choice(possible_gears)
                )
        return None

    def _get_rpm(self, gear: int, speed_kph: float) -> float:
        rpm = self.RPM_LOOKUP[gear] * speed_kph * 60
        return rpm

    def _generate_primary_filter(self, speed_kph: float) -> Dict:
        filter_dict = {
            "order": 1,
            "btype": "lowpass",
            "cutoff_hz": (
                self.primary_filter["speeddependence_cutoff_hzperkph"] * speed_kph
                + self.primary_filter["constant_hz"]
            )
            * np.random.choice(self.randomisationrange_freqmultiplier),
        }
        return filter_dict

    def _generate_secondary_filter(self, speed_kph: float) -> Dict:
        filter_dict = {
            "order": 2,
            "btype": "lowpass",
            "cutoff_hz": (
                self.secondary_filter["speeddependence_cutoff_hzperkph"] * speed_kph
                + self.secondary_filter["constant_hz"]
            )
            * np.random.choice(self.randomisationrange_freqmultiplier),
        }
        return filter_dict

    def _generate_bump_filter(self) -> Dict:
        bumpfilter_order = 1 if not self.random_flag else np.random.choice([1, 2])
        bumpfilter_f1_hz = (
            30 if not self.random_flag else np.random.randint(20, 101, size=1)[0]
        )
        bumpfilter_f2_hz = (
            60
            if not self.random_flag
            else bumpfilter_f1_hz * np.random.choice(np.arange(1.25, 2.01, 0.01))
        )
        filter_dict = {
            "order": int(bumpfilter_order),
            "btype": "bandpass",
            "cutoff_hz": [int(bumpfilter_f1_hz), int(bumpfilter_f2_hz)],
        }
        return filter_dict

    def _generate_dip_filter(self) -> Tuple[Dict, Dict]:
        dipfilter_order = 2 if not self.random_flag else np.random.choice([1, 2])
        dipfilter_f1_hz = (
            200 if not self.random_flag else np.random.choice(range(30, 201, 10))
        )
        dipfilter_f2_hz = (
            300 if not self.random_flag else dipfilter_f1_hz * np.random.uniform(2, 3)
        )
        filter_dict_low = {
            "order": int(dipfilter_order),
            "btype": "lowpass",
            "cutoff_hz": int(dipfilter_f1_hz),
        }
        filter_dict_high = {
            "order": int(dipfilter_order),
            "btype": "highpass",
            "cutoff_hz": int(dipfilter_f2_hz),
        }

        return filter_dict_low, filter_dict_high


if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    car_noise_parameters = CarNoiseParameters(random_flag=True)
    print(car_noise_parameters.gen_parameters(90))
