"""
Common machine learning models for time series forecasting and prediction.
"""

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor)
from xgboost import XGBRegressor
from ..utils.utils import ml_shape_repair, forecast_support, show_table
from ._base import BaseModelWrapper


class CommonMachineLearning(BaseModelWrapper):
    """
    Common Machine Learning models for time series forecasting and prediction.

    Machine Learning alias:
        - lr: Logistic regression
        - knn: KNN
        - svm: Support vector machine
        - dt: Decision tree
        - et: Extra tree
        - ada: AdaBoost
        - bag: Bagging
        - gb: Gradient boosting
        - rf: Random forest
        - xgb: XGBoost
    """
    name = 'CommonMachineLearning'

    def __init__(self, model_alias: str = 'lr', **kwargs):
        super().__init__(**kwargs)
        # Logistic regression
        if model_alias == 'lr':
            _model = LinearRegression(**kwargs)
        # KNN
        elif model_alias == 'knn':
            _model = KNeighborsRegressor(**kwargs)
        # Support vector machine
        elif model_alias == 'svm':
            _model = SVR(**kwargs)
        # Decision tree
        elif model_alias == 'dt':
            _model = DecisionTreeRegressor(**kwargs)
        # Extra tree
        elif model_alias == 'et':
            _model = ExtraTreeRegressor(**kwargs)
        # AdaBoost
        elif model_alias == 'ada':
            _model = AdaBoostRegressor(**kwargs)
        # Bagging
        elif model_alias == 'bag':
            _model = BaggingRegressor(**kwargs)
        # Gradient boosting
        elif model_alias == 'gb':
            _model = GradientBoostingRegressor(**kwargs)
        # Random forest
        elif model_alias == 'rf':
            _model = RandomForestRegressor(**kwargs)
        # XGBoost
        elif model_alias == 'xgb':
            _model = XGBRegressor(**kwargs)
        else:
            raise ValueError('Invalid model alias.')

        self.model_alias = model_alias
        self.model = _model
        self.name = self.model.__class__.__name__
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
        self.model = self.model.__class__(**self.model.get_params())
        self.name = self.model.__class__.__name__

    def get_params(self):
        return self.model.get_params()

    @staticmethod
    def get_ml_aliases():
        """
        Get machine learning aliases.
        """
        return ['lr', 'knn', 'svm', 'dt', 'et', 'ada', 'bag', 'gb', 'rf', 'xgb']
