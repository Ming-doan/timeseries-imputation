"""
Metrics for evaluating the performance of the model.
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_squared_error, r2_score


class Metrics:
    """
    Metrics for evaluating the performance of the model.
    """

    def __init__(self):
        self.metrics = pd.DataFrame(
            columns=['Model', 'Similarity', 'NMAE', 'RMSE', 'R2', 'FSD', 'FB', 'FA2'])

    @staticmethod
    def similarity(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Calculate the similarity between the actual and predicted output.
        """
        return np.sum(1 / (1 + np.abs(y_true - y_pred))) / y_true.shape[0]

    @staticmethod
    def nmae(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Calculate the Normalized Mean Absolute Error (NMAE).
        """
        return np.sum(np.abs(y_true - y_pred)) / y_true.shape[0]

    @staticmethod
    def r2(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Calculate the R2 score.
        """
        # y_true_mean = np.mean(y_true)
        # y_pred_mean = np.mean(y_pred)
        # return np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean)) / np.sqrt(np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
        return r2_score(y_true, y_pred)

    @staticmethod
    def rmse(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Calculate the Root Mean Squared Error (RMSE).
        """
        # return np.sqrt(np.sum((y_true - y_pred) ** 2) / y_true.shape[0])
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def fsd(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Calculate the Fraction of Standard Deviation (FSD).
        """
        y_true_sd = np.std(y_true)
        y_pred_sd = np.std(y_pred)
        return 2 * np.abs(y_true_sd - y_pred_sd) / (y_true_sd + y_pred_sd)

    @staticmethod
    def fb(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Calculate the Fraction of Bias (FB).
        """
        y_true_mean = np.mean(y_true)
        y_pred_mean = np.mean(y_pred)
        return 2 * (y_pred_mean - y_true_mean) / (y_pred_mean + y_true_mean)

    @staticmethod
    def fa2(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], upper_bound: float = 2.0, lower_bound: float = 0.5):
        """
        Calculate the Fraction of Absolute Error (FA2).
        """
        y = y_pred / y_true
        return np.where((y >= lower_bound) & (y <= upper_bound))[0].shape[0] / y_true.shape[0]

    def add_metrics(self, model: str, y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
        """
        Add metrics to the dataframe.
        """
        metrics = [model, Metrics.similarity(y_true, y_pred), Metrics.nmae(
            y_true, y_pred), Metrics.rmse(y_true, y_pred), Metrics.r2(y_true, y_pred), Metrics.fsd(y_true, y_pred), Metrics.fb(y_true, y_pred), Metrics.fa2(y_true, y_pred)]
        self.metrics.loc[len(self.metrics)] = metrics
        return metrics[1:]

    def save(self, path: str):
        """
        Save the metrics dataframe.
        """
        self.metrics.to_csv(path, index=False)
