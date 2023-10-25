"""
Long Short-Term Memory (LSTM) model.
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tqdm import tqdm
from ._base import BaseModelWrapper


class LongShortTermMemory(BaseModelWrapper):
    """
    Long Short-Term Memory (LSTM) model.
    """
    name = "LSTM"

    def __init__(self, n_features: int = 1, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.is_generator = True

        self.n_features = n_features
        self.epochs = kwargs.get('epochs', 100)
        self.early_stop = EarlyStopping(
            monitor='loss', patience=kwargs.get('patience', 3))
        self.optimizer = kwargs.get('optimizer', Adam(kwargs.get('lr', 0.001)))

        if layers is None:
            self.layers = [
                LSTM(units=50, return_sequences=True),
                LSTM(units=50),
                Dense(units=n_features)
            ]
        else:
            self.layers = layers

        self.model = None

    def fit(self, generator, x, y):
        self.model = Sequential([
            InputLayer(input_shape=(generator.window_size, self.n_features),
                       batch_size=generator.batch_size),
            *self.layers
        ])
        self.model.compile(optimizer=self.optimizer, loss="mse")
        self.model.fit(generator, epochs=self.epochs,
                       callbacks=[self.early_stop])

    def predict(self, generator, x):
        return self.model.predict(generator).squeeze()

    def forecast(self, x, steps):
        preds = []
        for _ in tqdm(range(steps), desc=f'Forecasting {self.name}'):
            preds.append(self.model.predict(x, verbose=0)[0])
        return np.array(preds).squeeze()

    def summary(self):
        # self.model.summary(print_fn=print)
        ...

    def reset(self):
        self.model.reset_states()