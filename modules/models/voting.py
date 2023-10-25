"""
Voting model for Time Series Forecasting.
"""

from sklearn.ensemble import VotingRegressor
from .mls import CommonMachineLearning
from ..utils.utils import ml_shape_repair, forecast_support, show_table
from ._base import BaseModelWrapper


class Voting(BaseModelWrapper):
    """
    Voting Regressor for Time Series Forecasting.
    """
    name = 'VotingRegressor'

    def __init__(self, models: list = None, **kwargs):
        super().__init__(**kwargs)
        aliases = CommonMachineLearning.get_ml_aliases()

        if models is None:
            self.models: list = [(
                alias, CommonMachineLearning(model_alias=alias).model) for alias in aliases]
        else:
            self.models = models
        self.model = VotingRegressor(estimators=self.models, **kwargs)

        self.is_generator = False

    def fit(self, generator, x, y):
        x, y = ml_shape_repair(x, y)
        # Fit the model
        self.model.fit(x, y)

    def predict(self, generator, x):
        x = ml_shape_repair(x)[0]
        # Predict the output
        return self.model.predict(x)

    def forecast(self, x, steps):
        # Forecast the output
        return forecast_support(self.model.predict, x.T, steps)

    def summary(self):
        m_params = self.model.get_params()
        show_table(data=[[self.name, *m_params.values()]],
                   cols=['Model', *m_params.keys()])

    def reset(self):
        self.model = VotingRegressor(estimators=self.models)
