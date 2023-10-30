"""
Trainer classes for Time Series Forecasting and Prediction.
"""
import time
from typing import Union
from loguru import logger
from .models._base import BaseModelWrapper
from .alias import get_by_alias
from .utils.generator import WindowGenerator
from .utils.callbacks import Callback
from .utils.metrics import Metrics
from .utils.cache import Cache


class Trainer:
    """
    Trainer Wrapper class for Time Series Forecasting and Prediction.
    """

    def __init__(self, model: Union[BaseModelWrapper, str, list], **kwargs):
        # Get model by alias or use the model directly
        if isinstance(model, list):
            if all(isinstance(m, str) for m in model):
                self.models = [get_by_alias(m, **kwargs) for m in model]
            elif all(isinstance(m, BaseModelWrapper) for m in model):
                self.models = model
            else:
                raise ValueError(
                    'Invalid model list. All models must be either string or subclass of BaseModelWrapper.')
        elif isinstance(model, str):
            self.models = [get_by_alias(model, **kwargs)]
        else:
            self.models = [model]

    def reset(self):
        """
        Reset all models.
        """
        for model in self.models:
            model.reset()
        logger.info('All models have been reset.')

    def train(self,
              train_generator: WindowGenerator,
              test_generator: WindowGenerator,
              callbacks: list[Callback] = None,
              cache: Cache = None):
        """
        Training the models.
        """
        # Steps:
        # 1. Fit the model
        # 2. Predict the output
        # 3. Forecast the output
        # 4. Run callbacks

        # Check train generator size
        if train_generator.window_size > len(train_generator.dataframe):
            raise ValueError(
                f'Invalid window size. window_size: {train_generator.window_size}, dataframe size: {len(train_generator.dataframe)}')
        if train_generator.batch_size > len(train_generator.dataframe):
            raise ValueError(
                f'Invalid batch size. batch_size: {train_generator.batch_size}, dataframe size: {len(train_generator.dataframe)}')

        # Check test generator size
        is_predict_able = True
        if test_generator.window_size >= len(test_generator.dataframe) - 1:
            is_predict_able = False
            logger.warning(
                f'Window size is larger than the dataframe size. {test_generator.window_size} >= {len(test_generator.dataframe) - 1}. Disabling prediction.')
        if test_generator.batch_size >= len(test_generator.dataframe) - 1:
            is_predict_able = False
            logger.warning(
                f'Batch size is larger than the dataframe size. {test_generator.batch_size} >= {len(test_generator.dataframe) - 1}. Disabling prediction.')

        # Iterate over all models
        for i, model in enumerate(self.models):
            ### Set the model for callbacks ###
            if callbacks is not None:
                for callback in callbacks:
                    callback.set_model(model)

            ### Show model summary ###
            logger.info(
                f'Model {i+1}/{len(self.models)}: {model.name}')
            model.summary()

            ### Fit the model ###
            logger.info(f'Fitting the model {model.name}')

            # Start the timer
            start_time = time.time()

            # If the model is not a generator, generate the data
            if not model.is_generator:
                # Cache implementation
                if cache is not None:
                    cache_id = str(id(train_generator.dataframe))
                    try:
                        x_train, y_train = cache.get(cache_id)
                        logger.info(f'Cache hit {cache_id}')
                    except KeyError as _:
                        logger.info(f'Cache miss {cache_id}')
                        cache.set(cache_id, train_generator.generate())
                        x_train, y_train = cache.get(cache_id)
                else:
                    x_train, y_train = train_generator.generate()

                # Fit the model when the data is generated
                model.fit(None, x_train, y_train)

            # Else, fit the model with the generator
            else:
                model.fit(train_generator, None, None)
            logger.info(
                f'Training completed in {time.time() - start_time:.2f}s')

            ### Predict the output ###
            if is_predict_able:
                # If the model is not a generator, generate the data
                if not model.is_generator:
                    # Cache implementation
                    if cache is not None:
                        cache_id = str(id(test_generator.dataframe))
                        try:
                            x_test, y_true = cache.get(cache_id)
                            logger.info(f'Cache hit {cache_id}')
                        except KeyError as _:
                            logger.info(f'Cache miss {cache_id}')
                            cache.set(cache_id, test_generator.generate())
                            x_test, y_true = cache.get(cache_id)
                    else:
                        x_test, y_true = test_generator.generate()

                    # Squeeze the y_true
                    y_true = y_true.squeeze()

                    # Predict the output when the data is generated
                    y_pred = model.predict(None, x_test)

                # Else, predict the output with the generator
                else:
                    y_pred = model.predict(test_generator, None)
                    y_true = test_generator.get_y_true()

                # Check the shape of the output
                if y_true.shape != y_pred.shape:
                    raise ValueError(
                        f'Invalid output shape. y_true: {y_true.shape}, y_pred: {y_pred.shape}')
                logger.info(
                    f'Similarity on predicting: {Metrics.similarity(y_true, y_pred)}')

                # Run callbacks
                if callbacks is not None:
                    for callback in callbacks:
                        callback.after_predict(y_true, y_pred)

            # Forecast the output
            y_fore = model.forecast(
                train_generator.get_last_window(), len(test_generator.dataframe))
            y_true = test_generator.dataframe.values.squeeze()
            if y_true.shape != y_fore.shape:
                raise ValueError(
                    f'Invalid output shape. y_true: {y_true.shape}, y_fore: {y_fore.shape}')
            logger.info(
                f'Similarity on forecasting: {Metrics.similarity(y_true, y_fore)}')

            # Run callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback.after_forecast(y_true, y_fore)

            logger.success(
                f'Model {i+1}/{len(self.models)}~{model.name} completed.')
