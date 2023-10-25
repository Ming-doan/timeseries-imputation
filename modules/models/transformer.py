"""
Transformer model for time series forecasting.
"""

from ._base import BaseModelWrapper


class TransformerTS(BaseModelWrapper):
    """
    Recurrent Network model for time series forecasting.
    """
    name = 'Transformer'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_generator = True

    def fit(self, generator, x, y):
        # If `is_generator` is True, the system will give generator as `WindowGenerator`.
        # x, and y are None
        # generator is a `WindowGenerator` object

        # If `is_generator` is False, the system will give x, and y as numpy arrays.
        # generator is None
        # x, and y is a numpy array of shape (data_length, window_size, n_features)

        # You can use `generator` in for loop to get batches of data.
        # >>> for data in generator:
        # >>>     print(data.shape) # (batch_size, window_size, n_features)
        ...

    def predict(self, generator, x):
        # If `is_generator` is True, the system will give generator as `WindowGenerator`.
        # x is None
        # generator is a `WindowGenerator` object

        # If `is_generator` is False, the system will give x as numpy arrays.
        # generator is None
        # x is a numpy array of shape (data_length, window_size, n_features)
        ...

    def forecast(self, x, steps):
        # x is a numpy array of shape (window_size, n_features)
        # steps is an integer. The number of future value to forecast.
        ...

    def summary(self):
        ...

    def reset(self):
        ...
