"""
Long Short-Term Memory (LSTM) model.
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Flatten, Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History
from ..utils.utils import forecast_support
from ._base import BaseModelWrapper


class LongShortTermMemory(BaseModelWrapper):
    """
    Long Short-Term Memory (LSTM) model.
    """
    name = "LSTM"

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
                LSTM(64, activation='relu', return_sequences=True),
                LSTM(32, activation='relu'),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(1)
            ]
        else:
            self.layers = layers

        self.model = None

    def fit(self, generator, x, y):
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
        return self.model.predict(generator).squeeze()

    def forecast(self, x, steps):
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
