"""
JeongStacking is a stacking model that is implemented by Jeong, J. (2021).
"""

from abc import abstractmethod
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor)
from xgboost import XGBRegressor
from ..utils.utils import ml_shape_repair, forecast_support
from ._base import BaseModelWrapper


class _BaseModel:
    @abstractmethod
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def fit(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]):
        """
        Fit the model.
        """

    @abstractmethod
    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Predict the output.
        """


class JeongStage:
    """
    Stage of JeongStacking.
    """

    def __init__(self, estimators: list[_BaseModel]):
        self.estimators = estimators


class JeongStackingRegressor(_BaseModel):
    """
    JeongStacking is a stacking model that is implemented by Jeong, J. (2021).
    """

    def __init__(self, stages: list[JeongStage], cv_splits: int = 5, random_state: int = 42):
        self.stages = self.__auto_format_stages(stages)
        self.cv_splits = cv_splits
        self.random_state = random_state
        self._x_train: npt.NDArray[np.float32] = None
        self._y_train: npt.NDArray[np.float32] = None
        self._x_new: npt.NDArray[np.float32] = None

        # Define KFold
        self.kfold = KFold(n_splits=cv_splits, shuffle=False)

        # Check if last estimator is more than 1
        if len(self.stages[-1].estimators) > 1:
            raise ValueError(
                f"Last stage must have only one estimator. Got {len(self.stages[-1].estimators)}.")

    def __auto_format_stages(self, stages: list[JeongStage] = None):
        formated_stages = []
        # Iterate over stages
        for stage in stages:
            # Check if stage is a list
            if isinstance(stage, list):
                formated_stages.append(JeongStage(estimators=stage))
            # Check if stage is a JeongStage
            elif isinstance(stage, JeongStage):
                formated_stages.append(stage)
            else:
                raise ValueError(
                    f"Invalid stage type. Must be list or JeongStage. Got {type(stage)}.")
        return formated_stages

    def __set_datasets(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]):
        # Check if shape of dataset is matched
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shape of x and y is not matched. Expected {x.shape[0]} but got {y.shape[0]}.")

        # Set dataset
        self._x_train = x
        self._y_train = y

    def __set_new_input(self, x: npt.NDArray[np.float32]):
        self._x_new = x

    def fit(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]):
        # Set initial dataset
        self.__set_datasets(x, y)

        # Show progress
        print('Fitting JeongStackingRegressor...')

        # Iterate over stages
        for i, stage in enumerate(self.stages):
            # Init cv predictions
            cv_predictions = None

            # Iterate over Cross Validation split
            for j, (train_index, test_index) in enumerate(self.kfold.split(self._x_train)):
                # Get train and test dataset
                x_train, x_test = self._x_train[train_index], self._x_train[test_index]
                y_train, _ = self._y_train[train_index], self._y_train[test_index]

                # Init stage predictions
                stage_predictions = None

                # Iterate over estimators
                for estimator in stage.estimators:
                    # Set progress description
                    print(
                        f'Stage {i+1}/{len(self.stages)}, Fold {j+1}/{self.cv_splits}, Estimator {estimator.__class__.__name__}', end='\r')

                    # Fit estimator
                    estimator.fit(x_train, y_train)
                    # Predict
                    y_pred = estimator.predict(x_test)

                    # Append to stage train predictions
                    if stage_predictions is None:
                        stage_predictions = y_pred.reshape(-1, 1)
                    else:
                        stage_predictions = np.append(
                            stage_predictions, y_pred.reshape(-1, 1), axis=1)

                # Append to cv train predictions
                if cv_predictions is None:
                    cv_predictions = stage_predictions
                else:
                    cv_predictions = np.append(
                        cv_predictions, stage_predictions, axis=0)
            # Set new dataset
            self.__set_datasets(cv_predictions, self._y_train)

    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Set new input
        self.__set_new_input(x)

        # Iterate over stages
        for stage in self.stages:
            # Init stage predictions
            stage_predictions = None

            # Iterate over estimators
            for estimator in stage.estimators:
                # Predict
                y_pred = estimator.predict(self._x_new)

                # Append to stage predictions
                if stage_predictions is None:
                    stage_predictions = y_pred.reshape(-1, 1)
                else:
                    stage_predictions = np.append(
                        stage_predictions, y_pred.reshape(-1, 1), axis=1)

            # Set new dataset
            self.__set_new_input(stage_predictions)

        return self._x_new.squeeze()


class JeongStacking(BaseModelWrapper):
    """
    JeongStacking is a stacking model that is implemented by Jeong, J. (2021).
    """
    name = 'JeongStacking'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stage: list[JeongStage] = [JeongStage(estimators=[
                    LinearRegression(),
                    Ridge(),
                    RidgeCV(),

                    SGDRegressor(),

                    KNeighborsRegressor(n_neighbors=7),
                    KNeighborsRegressor(n_neighbors=5, metric='l1'),
                    KNeighborsRegressor(n_neighbors=3, metric='l2'),
                    KNeighborsRegressor(),

                    RadiusNeighborsRegressor(radius=3),
                    RadiusNeighborsRegressor(radius=5),
                    RadiusNeighborsRegressor(radius=7),
                    RadiusNeighborsRegressor(),


                    SVR(kernel="linear", C=100, gamma="auto", degree=4, epsilon=0.1, coef0=1),
                    SVR(kernel="linear", C=50, gamma="auto", degree=3, epsilon=0.3, coef0=1),
                    SVR(kernel="linear", C=200, gamma="auto", degree=2, epsilon=0.2, coef0=1),
                    SVR(),

                    DecisionTreeRegressor(max_depth=5),
                    DecisionTreeRegressor(max_depth=3),
                    DecisionTreeRegressor(max_depth=7),
                    DecisionTreeRegressor(),


                    ExtraTreeRegressor(max_depth=5),
                    ExtraTreeRegressor(max_depth=3),
                    ExtraTreeRegressor(max_depth=7),
                    ExtraTreeRegressor(),


                    AdaBoostRegressor(KNeighborsRegressor(n_neighbors=5)),
                    AdaBoostRegressor(DecisionTreeRegressor(max_depth=4)),
                    AdaBoostRegressor(ExtraTreeRegressor()),
                    AdaBoostRegressor(SVR()),
                    AdaBoostRegressor(),

                    BaggingRegressor(KNeighborsRegressor(n_neighbors=5)),
                    BaggingRegressor(DecisionTreeRegressor(max_depth=4)),
                    BaggingRegressor(ExtraTreeRegressor()),
                    BaggingRegressor(SVR()),
                    BaggingRegressor(),

                    GradientBoostingRegressor(learning_rate= 0.1, n_estimators= 100, subsample = 1, criterion= "friedman_mse", min_samples_split = 2, min_samples_leaf = 1, max_depth = 5),
                    GradientBoostingRegressor(learning_rate= 0.01, n_estimators= 200, subsample = 1, criterion= "friedman_mse", min_samples_split = 2, min_samples_leaf = 2, max_depth = 5),
                    GradientBoostingRegressor(learning_rate= 0.1, n_estimators= 50, subsample = 1, criterion= "friedman_mse", min_samples_split = 2, min_samples_leaf = 1, max_depth = 3),
                    GradientBoostingRegressor(),


                    RandomForestRegressor(max_depth=3),
                    RandomForestRegressor(max_depth=5),
                    RandomForestRegressor(max_depth=7),
                    RandomForestRegressor(),


                    HistGradientBoostingRegressor(learning_rate=0.01, max_depth=5),
                    HistGradientBoostingRegressor(learning_rate=0.1, max_depth=7),
                    HistGradientBoostingRegressor(learning_rate=0.01, max_depth=3),
                    HistGradientBoostingRegressor(),


                    XGBRegressor()
                ]),

            JeongStage(estimators=[
                    Ridge(),
                    KNeighborsRegressor(n_neighbors=5),
                    SVR(),
                    DecisionTreeRegressor(max_depth=5),
                    AdaBoostRegressor(),
                    BaggingRegressor(),
                    RandomForestRegressor(),
                    XGBRegressor()
                ]),
            
             JeongStage(estimators=[LinearRegression()])
        ]
        self.model = JeongStackingRegressor(
            stages=kwargs.get('stages', self.stage))

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
        print('JeongStackingRegressor:')
        print('-----------------------')
        for i, stage in enumerate(self.model.stages):
            print(f'- Stage {i+1}:')
            for estimator in stage.estimators:
                model_params = str(estimator.get_params()).replace(
                    "{", "").replace("}", "").replace("'", "")
                print(
                    f'    - {estimator.__class__.__name__} ({model_params})')

    def reset(self):
        self.model = JeongStackingRegressor(stages=self.model.stages)

    def get_params(self):
        params = {}
        for i, stage in enumerate(self.stage):
            params[f'stage_{i+1}'] = {}
            for estimator in stage.estimators:
                params[f'stage_{i+1}'][estimator.__class__.__name__] = estimator.get_params()
        return params
