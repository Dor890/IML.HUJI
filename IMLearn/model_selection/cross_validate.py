from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.metrics.loss_functions import mean_square_error


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator
    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data
    X: ndarray of shape (n_samples, n_features)
       Input data to fit
    y: ndarray of shape (n_samples, )
       Responses of input data to fit to
    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.
    cv: int
        Specify the number of folds.
    Returns
    -------
    train_score: float
        Average train score over folds
    validation_score: float
        Average validation score over folds
    """
    train_score, validation_score = 0, 0
    for i in range(cv):
        folds_arr = np.arange(y.size) % cv  # Sets the partition
        train_X, train_y = X[folds_arr != i], y[folds_arr != i]
        validate_X, validate_y = X[folds_arr == i], y[folds_arr == i]
        estimator.fit(train_X, train_y)
        train_score += scoring(train_y, estimator.predict(train_X))
        validation_score += scoring(validate_y, estimator.predict(validate_X))
    return train_score / cv, validation_score / cv

