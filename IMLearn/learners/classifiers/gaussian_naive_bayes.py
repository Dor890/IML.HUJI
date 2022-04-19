from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import accuracy


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
        # np.vsplit - separate mean by class
        self.classes_ = np.unique(y)
        m = X.shape[0]
        self.mu_ = np.array(
            [np.mean(X[y == cur_class], axis=0) for cur_class in
             self.classes_])
        self.pi_ = [(1/m)*np.sum([1 for i in y if i == cur_class]) for cur_class in self.classes_]
        self.vars_ = np.array(
            [np.var(X[y == cur_class], axis=0) for cur_class in
             self.classes_])

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
            cov = np.diag(self.vars_[k])
            inv_cov = np.linalg.inv(cov)
            each_sample = []
            for X_j in X:
                a_k = np.dot(inv_cov, self.mu_[k])
                b_k = np.log(self.pi_[k]) - 0.5 * \
                      np.matmul(np.matmul(self.mu_[k], inv_cov),
                                self.mu_[k])
                each_sample.append(np.matmul(a_k.T, X_j.T)+b_k)
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
        # total_arr = []
        # first = 1 / (np.sqrt(np.power(2 * np.pi, X.shape[1])) * det(self.cov_))
        # for x in X:
        #     cur_arr = []
        #     for k in self.classes_:
        #         mahalanobis = np.matmul(np.matmul(
        #             x-self.mu_[k], self._cov_inv), x-self.mu_[k])
        #         full = np.exp(-0.5 * mahalanobis) / first * self.pi_[k]
        #         cur_arr.append(full)
        #     total_arr.append(cur_arr)
        # return np.array(total_arr)

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
