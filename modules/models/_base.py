"""
Base model used for Time Series Forecasting and Prediction.
"""

from abc import abstractmethod
from typing import Union
import numpy as np
import numpy.typing as npt
from ..utils.generator import WindowGenerator


class BaseModelWrapper:
    """
    Base model used for Time Series Forecasting and Prediction.
    """
    name = 'BaseModelWrapper'

    def __init__(self, **kwargs):
        self.is_generator = True
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, generator: WindowGenerator, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]):
        """
        Fit the model to the training data.
        """

    @abstractmethod
    def predict(self, generator: WindowGenerator, x: npt.NDArray[np.float32]) -> Union[npt.NDArray[np.float32], tuple[npt.NDArray[np.float32]]]:
        """
        Predict the output for the given input.
        Return shape: `(n, )`. n is the number of samples in the generator.
        """

    @abstractmethod
    def forecast(self, x: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        """
        Forecast the output for the given input.
        Return shape: `(steps, )`.
        """

    @abstractmethod
    def summary(self):
        """
        Print the summary of the model.
        """

    @abstractmethod
    def reset(self):
        """
        Reset the model.
        """

    @abstractmethod
    def get_params(self):
        """
        Get the parameters of the model.
        """
