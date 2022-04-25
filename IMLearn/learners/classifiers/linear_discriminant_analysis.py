from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import accuracy


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        k = self.classes_.size
        m = X.shape[0]
        self.mu_ = np.array([np.mean(X[y == cur_class], axis=0)
                             for cur_class in self.classes_])
        mu_y_ = np.array([self.mu_[int(y_i)] for y_i in y])
        cov_helper = np.array([X[i] - mu_y_[i] for i in range(m)])
        self.cov_ = np.matmul(cov_helper.T, cov_helper) / (m-k)  # Unbiased estimator
        try:
            self._cov_inv = inv(self.cov_)
        except np.linalg.LinAlgError:
            self._cov_inv = self.cov_
        self.pi_ = np.array([(1/m)*np.sum([1 for i in y if i == cur_class])
                             for cur_class in self.classes_])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        each_class = []
        for k in range(self.classes_.size):
            each_sample = []
            for X_j in X:
                a_k = np.dot(self._cov_inv, self.mu_[k])
                b_k = np.log(self.pi_[k]) - 0.5 * \
                      np.matmul(np.matmul(self.mu_[k], self._cov_inv),
                                self.mu_[k])
                each_sample.append(np.matmul(a_k.T, X_j) + b_k)
            each_class.append(each_sample)
        each_class = np.transpose(each_class)
        pred_class = [np.argmax(each_class[i]) for i in range(X.shape[0])]
        predicted = np.array([self.classes_[i] for i in pred_class])
        return predicted

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        total_arr = []
        first = 1 / (np.sqrt(np.power(2*np.pi, X.shape[1])) * det(self.cov_))
        for X_j in X:
            cur_arr = []
            for k in self.classes_:
                mahalanobis = np.matmul(np.matmul(
                    X_j - self.mu_[k], self._cov_inv), X_j - self.mu_[k])
                full = np.exp(-0.5 * mahalanobis) / first * self.pi_[k]
                cur_arr.append(full)
            total_arr.append(cur_arr)
        return np.array(total_arr)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return accuracy(y, self.predict(X))
