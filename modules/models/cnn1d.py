"""
CNN1D model for time series forecasting.
"""

from ._base import BaseModelWrapper
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tqdm import tqdm


class CNN1D(BaseModelWrapper):
    """
    CNN1D model for time series forecasting.
    """
    name = 'CNN1D'

    def __init__(self, n_features: int = 1, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.is_generator = True

        self.n_features = n_features
        self.epochs = kwargs.get('epochs', 200)
        self.early_stop = EarlyStopping(
            monitor='loss', patience=kwargs.get('patience', 3))
        self.optimizer = kwargs.get('optimizer', Adam(kwargs.get('lr', 0.001)))

        if layers is None:
            self.layers = [
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(1)
            ]
        else:
            self.layers = layers

        self.model = None

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
        self.model = Sequential([
            InputLayer(input_shape=(generator.window_size, self.n_features),
                       batch_size=generator.batch_size),
            *self.layers
        ])
        self.model.compile(optimizer=self.optimizer, loss="mse")
        # Show model summary
        self.model.summary()
        # Fit model
        self.model.fit(generator, epochs=self.epochs,
                       callbacks=[self.early_stop])

    def predict(self, generator, x):
        # If `is_generator` is True, the system will give generator as `WindowGenerator`.
        # x is None
        # generator is a `WindowGenerator` object

        # If `is_generator` is False, the system will give x as numpy arrays.
        # generator is None
        # x is a numpy array of shape (data_length, window_size, n_features)
        return self.model.predict(generator).squeeze()

    def forecast(self, x, steps):
        # x is a numpy array of shape (window_size, n_features)
        # steps is an integer. The number of future value to forecast.
        preds = []
        x = x.reshape(1, *x.shape)
        for _ in tqdm(range(steps), desc=f'Forecasting {self.name}'):
            preds.append(self.model.predict(x, verbose=0)[0])
        return np.array(preds).squeeze()

    def summary(self):
        print(f'{self.name} model summary:')

    def reset(self):
        self.model.reset_states()
