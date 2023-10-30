"""
Custom Dataset generator for Time Series Forecasting and Prediction.
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
from keras.utils import Sequence
from tqdm import tqdm


class WindowGenerator(Sequence):
    """
    Data set generator.
    This dataset is not contained any missing value. 
    The generator only create the window of the data.

    >>> generator = WIndowGenerator(dataframe, window_size=10, batch_size=32)
    >>> generator[0] # Get the first batch
    >>> generator.generate() # Get all batches

    Output shape: 
        X: `(batch_size, window_size, n_features)`\n
        y: `(batch_size, n_features)`
    """

    def __init__(self, dataframe: pd.DataFrame, window_size: int, batch_size: int = 1, selected_cols: list[str] = None):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('Invalid dataframe. Must be pandas.DataFrame.')
        if selected_cols is not None:
            if not isinstance(selected_cols, list):
                raise ValueError(
                    'Invalid selected columns. Must be list of string.')
            self.dataframe = dataframe[selected_cols]
        else:
            self.dataframe = dataframe

        if not isinstance(window_size, int):
            raise ValueError('Invalid window size. Must be integer.')
        self.window_size = window_size

        if not isinstance(batch_size, int):
            raise ValueError('Invalid batch size. Must be integer.')
        if batch_size < 1 or batch_size > len(dataframe):
            raise ValueError(
                'Invalid batch size. Must be greater than 0 and less than the length of the dataframe.')
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.dataframe) - self.window_size) // self.batch_size

    def __getitem__(self, index) -> npt.NDArray[np.float32]:
        x = []
        y = []
        for i in range(self.batch_size):
            _idx = self.batch_size * index + i
            x.append(self.dataframe.iloc[_idx:_idx + self.window_size].values)
            y.append(self.dataframe.iloc[_idx + self.window_size].values)
        return np.array(x), np.array(y)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def generate(self) -> tuple[npt.NDArray[np.float32]]:
        """
        Generete all windows.
        """
        batch_size = self.batch_size
        # Assign self.batch_size to 1 to get all windows
        self.batch_size = 1
        x = None
        y = None
        progress = tqdm(range(len(self)), desc='Generating windows')
        for i in progress:
            _x, _y = self[i]
            if x is None:
                x = _x
                y = _y
            else:
                x = np.concatenate((x, _x))
                y = np.concatenate((y, _y))
        # Assign self.batch_size to original value
        self.batch_size = batch_size
        return np.array(x), np.array(y)

    def get_last_window(self) -> npt.NDArray[np.float32]:
        """
        Get the last window.
        """
        return self.dataframe.iloc[-self.window_size:].values

    def get_y_true(self) -> npt.NDArray[np.float32]:
        """
        Get True values of dataset.
        """
        return self.dataframe.iloc[self.window_size:self.window_size + (len(self) * self.batch_size)].values.squeeze()
