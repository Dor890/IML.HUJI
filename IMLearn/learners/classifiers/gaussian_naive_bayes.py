from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import accuracy
from numpy.linalg import det, inv


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        m, f = X.shape[0], X.shape[1]
        self.mu_ = np.array([np.mean(X[y == cur_class], axis=0)
                             for cur_class in self.classes_])
        self.pi_ = np.array([(1/m)*np.sum([1 for i in y if i == cur_class])
                    for cur_class in self.classes_])
        self.vars_ = np.array([np.var(X[y == cur_class], axis=0, ddof=1)
                               for cur_class in self.classes_])  # Unbiased

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
        predicted = []
        for X_j in X:
            each_sample = []
            for k in self.classes_:
                inv_cov = inv(np.diag(self.vars_[k]))
                diff = X_j - self.mu_[k]
                likelihood = 0.5 * np.log(det(inv_cov)) - \
                             0.5 * diff.T @ inv_cov @ diff
                posterior = np.log(self.pi_[k]) + likelihood
                each_sample.append(posterior)
            pred_class = self.classes_[np.argmax(each_sample)]
            predicted.append(pred_class)
        return np.array(predicted)


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
        likelihood_arr = []
        for X_j in X:
            each_sample = []
            for k in self.classes_:
                inv_cov = inv(np.diag(self.vars_[k]))
                diff = X_j - self.mu_[k]
                likelihood = 0.5 * np.log(det(inv_cov)) - \
                             0.5 * diff.T @ inv_cov @ diff
                each_sample.append(likelihood)
            likelihood_arr.append(each_sample)
        return np.array(likelihood_arr)

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
