import numpy as np

from clarity.enhancer.multiband_compressor.crossover.crossover_one import CrossoverOne
from clarity.enhancer.multiband_compressor.crossover.crossover_two import (
    CrossoverTwoOrMore,
)


class Crossover:
    """
    Emulate a multiple constructors of the Crossover class to facilitate the use.
    """

    def __init__(self, *args):
        if isinstance(args[0], float) or isinstance(args[0], int):
            self.crossover = CrossoverOne(*args)
        elif isinstance(args[0], list) or isinstance(args[0], np.ndarray):
            if len(args[0]) == 1:
                self.crossover = CrossoverOne(*args)
            else:
                self.crossover = CrossoverTwoOrMore(*args)
        else:
            raise ValueError("Invalid arguments.")

    def __call__(self, *args, **kwargs):
        return self.crossover(*args, **kwargs)


__all__ = ["Crossover"]
