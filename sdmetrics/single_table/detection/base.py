"""Base class for Machine Learning Detection metrics for single table datasets."""

import logging

import numpy as np
from rdt import HyperTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric

LOGGER = logging.getLogger(__name__)


class DetectionMetric(SingleTableMetric):
    """Base class for Machine Learning Detection based metrics on single tables.

    These metrics build a Machine Learning Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'SingleTable Detection'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _fit_predict(X_train, y_train, X_test):
        """Fit a classifier and then use it to predict."""
        raise NotImplementedError()

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, dtypes=None):
        """Compute this metric.

        This builds a Machine Learning Classifier that learns to tell the synthetic
        data apart from the real data, which later on is evaluated using Cross Validation.

        The output of the metric is one minus the average ROC AUC score obtained.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.

        Returns:
            float:
                One minus the ROC AUC Cross Validation Score obtained by the classifier.
        """
        metadata = cls._validate_inputs(real_data, synthetic_data, metadata)

        transformer = HyperTransformer(dtype_transformers={'O': 'one_hot_encoding'}, dtypes=dtypes)
        real_data = transformer.fit_transform(real_data).values
        synthetic_data = transformer.transform(synthetic_data).values

        X = np.concatenate([real_data, synthetic_data])
        y = np.hstack([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
        if np.isin(X, [np.inf, -np.inf]).any():
            X[np.isin(X, [np.inf, -np.inf])] = np.nan

        try:
            scores = []
            kf = StratifiedKFold(n_splits=3, shuffle=True)
            for train_index, test_index in kf.split(X, y):
                y_pred, clf = cls._fit_predict(X[train_index], y[train_index], X[test_index])
                roc_auc = roc_auc_score(y[test_index], y_pred)
                scores.append(max(0.5, roc_auc) * 2 - 1)

            plot = False
            if plot:
                fpr, tpr, _ = roc_curve(y[test_index], y_pred)
                dummy_fpr = np.linspace(0, 1)
                dummy_tpr = np.linspace(0, 1)
                # plot the roc curve for the model
                plt.plot(dummy_fpr, dummy_tpr, linestyle='--', label="Random Classifier")
                plt.plot(fpr, tpr, marker=',', label='ROC-curve')
                plt.fill_between(dummy_tpr, tpr)
                # axis labels
                plt.title("ROC-Curve Churn")
                plt.xlabel('False-Positive Rate')
                plt.ylabel('True-Positive Rate')
                # show the legend
                plt.legend()
                # show the plot
                plt.show()

            return 1 - np.mean(scores)

        except ValueError as err:
            LOGGER.info('DetectionMetric: Skipping due to %s', err)
            return np.nan


