from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 5)
    print("Univariate samples are:")
    print(samples)
    uni_gaussian = UnivariateGaussian()
    uni_gaussian.fit(samples)
    print("Estimated (expectation, variance):")
    print((uni_gaussian.mu_, uni_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent

    # Question 3 - Plotting Empirical PDF of fitted model


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 0, 4])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, sigma, 5)
    print("Multivariate samples are:")
    print(samples)
    multi_gaussian = MultivariateGaussian()
    multi_gaussian.fit(samples)
    print("Estimated expectation vector:")
    print(multi_gaussian.mu_)
    print("Estimated covariance matrix:")
    print(multi_gaussian.cov_)

    # Question 5 - Likelihood evaluation

    # Question 6 - Maximum likelihood


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
