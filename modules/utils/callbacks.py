"""
Callbacks function for training.
"""

from abc import abstractmethod
import enum
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from loguru import logger
from ..models._base import BaseModelWrapper
from .missing import CreateMissingDataFrame
from .metrics import Metrics


plt.style.use('ggplot')


class Callback:
    """
    Base class used to build new callbacks.
    """

    def __init__(self):
        self.model: BaseModelWrapper = None

    def set_model(self, model: BaseModelWrapper):
        """
        Set the model for the callback.
        """
        self.model = model

    @abstractmethod
    def after_predict(self, y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Run after predicting the output.
        """

    @abstractmethod
    def after_forecast(self, y_true: npt.NDArray[np.float32], y_fore: npt.NDArray[np.float32]):
        """
        Run after forecasting the output.
        """


class SavePlot(Callback):
    """
    Save comparison plot between the actual and predicted or forecasted output.
    """

    def __init__(self, n_models: int, save_directory: str = None):
        super().__init__()
        self.save_directory = save_directory
        self.n_times = n_models * 2
        # Tracking
        self.tracking = 0
        self.phase = 1

    def after_predict(self, y_true, y_pred):
        """
        Save the plot after predicting the output.
        """

        plt.figure(figsize=(15, 5))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Predicted vs Actual of {self.model.name}')
        plt.legend()
        plt.show()
        plt.close()

    def after_forecast(self, y_true, y_fore):
        """
        Save the plot after forecasting the output.
        """
        # Increase the tracking
        self.tracking += 1

        # Direction of pipeline
        direction = 'Pipeline' if self.tracking <= self.n_times // 2 else 'Reverse_Pipeline'

        plt.figure(figsize=(15, 5))
        plt.plot(y_true, label='Actual')
        plt.plot(y_fore, label='Forecasted')
        plt.title(f'Forecasted vs Actual of {self.model.name} on {direction}')
        plt.legend()
        if self.save_directory is not None:
            plt.savefig(self.save_directory +
                        f'/{self.model.name}_{direction}_{self.phase}.png')
        plt.show()
        plt.close()

        # Increment the phase
        if self.tracking == self.n_times:
            self.phase += 1
            self.tracking = 0


class CombinationMode(enum.Enum):
    """
    Combination mode for combining the pipeline and reverse pipeline.
    """
    MEAN = 'mean'
    DATA_PER = 'data_per'
    SIMILARITY = 'similarity'

    @staticmethod
    def get_combination_mode(mode: str):
        """
        Get the combination mode.
        """
        if mode == CombinationMode.MEAN.value:
            return CombinationMode.MEAN
        elif mode == CombinationMode.DATA_PER.value:
            return CombinationMode.DATA_PER
        elif mode == CombinationMode.SIMILARITY.value:
            return CombinationMode.SIMILARITY
        else:
            raise ValueError(
                f'Invalid combination mode: {mode}. Please choose from {CombinationMode}')


class Combined(Callback):
    """
    Combine pipeline and reverse pipeline.
    """

    def __init__(self, n_models: int, combination_mode: str = CombinationMode.DATA_PER, df: CreateMissingDataFrame = None, save_directory: str = None):
        super().__init__()
        # Override model
        self.model: list[BaseModelWrapper] = []
        self.n_times = n_models * 2
        self.save_directory = save_directory
        # Arguments for combining
        self.actual = None
        self.pipeline = []
        self.reverse_pipeline = []
        self.combination_mode = combination_mode if isinstance(
            combination_mode, CombinationMode) else CombinationMode.get_combination_mode(combination_mode)
        # Window generator
        self.pipeline_data_length, self.reverse_pipeline_data_length = self.calculate_data_length(
            df)
        # Metrics
        self.metrics = Metrics()
        # Tracking
        self.n_times_tracking = 0
        self.phase = 1

    def calculate_data_length(self, creator: CreateMissingDataFrame):
        """
        Calculate the length of the data.
        """
        if creator is None:
            return None, None
        data_length = len(creator.dataframe)
        pipeline_data_length = creator.missing_indexs[0][0]
        reverse_pipeline_data_length = data_length - \
            creator.missing_indexs[0][1]
        return pipeline_data_length, reverse_pipeline_data_length

    def set_model(self, model: BaseModelWrapper):
        self.model.append(model)

    def after_predict(self, y_true, y_pred): ...

    def after_forecast(self, y_true, y_fore):
        # Increment the tracking
        self.n_times_tracking += 1
        # Save y_true as actual
        if self.actual is None:
            self.actual = y_true
        # Save y_fore as pipeline
        if len(self.pipeline) == len(self.reverse_pipeline):
            self.pipeline.append(y_fore)
        # Save y_fore as reverse pipeline
        else:
            self.reverse_pipeline.append(y_fore[::-1])

        if self.n_times_tracking == self.n_times:
            for i, pipe in enumerate(self.pipeline):
                # Get reverse pipeline
                reverse_pipe = self.reverse_pipeline[i]
                # Get model name
                model_name = self.model[i].name
                # When the reverse pipeline is saved, combine the pipeline and reverse pipeline
                if self.combination_mode == CombinationMode.MEAN:
                    combined_pipeline = np.mean(
                        (pipe, reverse_pipe), axis=0)
                elif self.combination_mode == CombinationMode.DATA_PER:
                    if self.pipeline_data_length is None or self.reverse_pipeline_data_length is None:
                        raise ValueError(
                            'Create Missing Dataframe `df` is required to calculate the data length')
                    # Calculate the percentage of data
                    alpha = self.pipeline_data_length / \
                        (self.pipeline_data_length +
                         self.reverse_pipeline_data_length)
                    # Combine the pipeline and reverse pipeline
                    combined_pipeline = alpha * pipe + \
                        (1 - alpha) * reverse_pipe
                elif self.combination_mode == CombinationMode.SIMILARITY:
                    # Calculate similarity
                    pipe_similarity = Metrics.similarity(self.actual, pipe)
                    reverse_pipe_similarity = Metrics.similarity(
                        self.actual, reverse_pipe)
                    alpha = pipe_similarity / \
                        (pipe_similarity + reverse_pipe_similarity)
                    # Combine the pipeline and reverse pipeline
                    combined_pipeline = alpha * pipe + \
                        (1 - alpha) * reverse_pipe
                else:
                    raise ValueError(
                        f'Invalid combination mode: {self.combination_mode}. Please choose from {CombinationMode}')

                # Save the plot
                plt.figure(figsize=(15, 5))
                plt.plot(self.actual, label='Actual')
                plt.plot(combined_pipeline, label='Combined Pipeline')
                plt.title(
                    f'Combined Pipeline vs Actual of {model_name}')
                plt.legend()
                if self.save_directory is not None:
                    plt.savefig(self.save_directory +
                                f'/{model_name}_combined_{self.phase}.png')
                plt.show()
                plt.close()

                # Calculate metrics
                metrics = self.metrics.add_metrics(
                    model_name, self.actual, combined_pipeline)

                logger.info(
                    f'Similarity on combining of model {model_name}: {metrics[0]}')

            # Increment the phase
            self.phase += 1
            # Reset the tracking
            self.n_times_tracking = 0
            self.model = []
            # Reset the pipeline
            self.pipeline = []
            self.reverse_pipeline = []
            self.actual = None
