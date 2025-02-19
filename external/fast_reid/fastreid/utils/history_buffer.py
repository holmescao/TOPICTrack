

import numpy as np
from typing import List, Tuple


class HistoryBuffer:


    def __init__(self, max_length: int = 1000000):
 
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = [] 
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: float = None):
  
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self):
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int):
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int):
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self):
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self):
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data
