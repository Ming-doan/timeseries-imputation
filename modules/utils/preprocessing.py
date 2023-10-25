"""
Preprocessing of the data
"""

from abc import abstractmethod
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler
from loguru import logger


class Plugins:
    """
    Base class for all plugins.
    """

    def __init__(self): ...

    @abstractmethod
    def flow(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Flow the input through the plugin.
        """

    @abstractmethod
    def reverse_flow(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Reverse flow the input through the plugin.
        """


class OutlierRemoval(Plugins):
    """
    Remove outliers from the data.
    """

    def __init__(self, upper_bound: float = 0.75, lower_bound: float = 0.25, shift: float = 3.0):
        super().__init__()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.shift = shift

    def flow(self, x):
        q1 = np.quantile(x, self.lower_bound)
        q3 = np.quantile(x, self.upper_bound)
        iqr = q3 - q1
        outlier = (x < q1 - self.shift * iqr) | (x > q3 + self.shift * iqr)
        x[outlier] = np.mean(x[~outlier])
        return x

    def reverse_flow(self, x):
        return x


class Scaler(Plugins):
    """
    Scale the data.
    """

    def __init__(self, scaler: MinMaxScaler = None):
        super().__init__()
        if scaler is None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler

    def flow(self, x):
        return self.scaler.fit_transform(x)

    def reverse_flow(self, x):
        return self.scaler.inverse_transform(x)


class Preprocessing:
    """
    Preprocessing of the data
    """

    def __init__(self):
        self.plugins: list[Plugins] = [
            OutlierRemoval(),
            Scaler()
        ]

    def flow(self, x: pd.DataFrame):
        """
        Flow the input through the plugins.
        """
        if not isinstance(x, pd.DataFrame):
            raise ValueError('Input must be a pandas DataFrame.')

        # Convert to numpy array
        _x = x.values

        for plugin in self.plugins:
            _x = plugin.flow(_x)
        return pd.DataFrame(_x, columns=x.columns)

    def reverse_flow(self, x: pd.DataFrame):
        """
        Reverse flow the input through the plugins.
        """
        if not isinstance(x, pd.DataFrame):
            raise ValueError('Input must be a pandas DataFrame.')

        # Convert to numpy array
        _x = x.values

        for plugin in self.plugins:
            _x = plugin.reverse_flow(_x)
        return pd.DataFrame(_x, columns=x.columns)

    def add_plugin(self, plugin: Plugins):
        """
        Add a plugin to the preprocessing pipeline.
        """
        self.plugins.append(plugin)

    def show(self):
        """
        Show the plugins used in the pipeline.
        """
        logger.info(f'Plugins: {[p.__class__.__name__ for p in self.plugins]}')
