"""
Recurrent Neural Network (RNN) model.
"""

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, InputLayer, Flatten, Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History
from ..utils.utils import forecast_support
from ._base import BaseModelWrapper


class RecurrentNetwork(BaseModelWrapper):
    """
    Recurrent Neural Network (RNN) model.
    """
    name = 'RNN'

    def __init__(self, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.is_generator = True

        self.epochs = kwargs.get('epochs', 100)
        self.early_stop = EarlyStopping(
            monitor='loss', patience=kwargs.get('patience', 3))
        self.histories = []
        self.optimizer = kwargs.get('optimizer', Adam(kwargs.get('lr', 0.001)))

        if layers is None:
            self.layers: list[Layer] = [
                SimpleRNN(64, activation='relu', return_sequences=True),
                SimpleRNN(32, activation='relu'),
                Flatten(),
                Dense(128, activation='relu'),
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
            InputLayer(input_shape=(generator.window_size, generator.dataframe.shape[1]),
                       batch_size=generator.batch_size),
            *self.layers
        ])
        self.model.compile(optimizer=self.optimizer, loss="mse")
        # Show model summary
        self.model.summary()
        # Fit model
        history = History()
        self.model.fit(generator, epochs=self.epochs,
                       callbacks=[self.early_stop, history])
        self.histories.append(history)

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
        return forecast_support(self.model.predict, x.reshape(1, -1), steps, verbose=0)

    def summary(self):
        print(f'{self.name} model summary:')

    def reset(self):
        self.model.reset_states()

    def get_params(self):
        params = {}
        for layer in self.layers:
            params[layer.name] = layer.get_config()
        return params
