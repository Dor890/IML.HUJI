from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree
    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate
    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    response = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    X = np.random.uniform(-3, 3, n_samples)
    eps = np.random.normal(0, noise, size=n_samples)
    y_noiseless = response(X)
    train_X, train_y, test_X, test_y = split_train_test(
        pd.DataFrame(X), pd.Series(y_noiseless), 2/3)
    fig1 = go.Figure([go.Scatter(x=train_X.to_numpy().flatten(), y=train_y, mode='markers',
                                name="Train Data",
                                marker=dict(color="black", opacity=0.7)),
                     go.Scatter(x=test_X.to_numpy().flatten(), y=test_y, mode='markers',
                                name="Test Data",
                                marker=dict(color="orange", opacity=0.7))],
                    layout=go.Layout(
                        title="True (noiseless) Model For Function f(x)=(x+3)(x+2)(x+1)(x-1)(x-2),"
                              " noise = {}, samples = {}".format(noise, n_samples),
                        xaxis_title="Data",
                        yaxis_title="Response",
                        height=400))
    fig1.show()
    fig1.write_image("ex5/true_model_noise_{}_samples_{}.png".format(noise, n_samples))

    y_noise = y_noiseless + eps
    train_X, train_y, test_X, test_y = split_train_test(
        pd.DataFrame(X), pd.Series(y_noise), 2/3)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    poly_deg = 10
    train_scores, validate_scores = np.zeros(poly_deg), np.zeros(poly_deg)
    train_X, train_y = train_X.to_numpy().flatten(), train_y.to_numpy()
    for k in range(poly_deg):
        train_scores[k], validate_scores[k] = \
            cross_validate(estimator=PolynomialFitting(k), X=train_X, y=train_y,
                           scoring=mean_square_error)
    fig2 = go.Figure([go.Scatter(x=[i for i in range(poly_deg)], y=train_scores,
                                 opacity=0.75, marker_color="black", name="Train Errors"),
                     go.Scatter(x=[i for i in range(poly_deg)], y=validate_scores,
                                 opacity=0.75, marker_color="orange", name="Validate Errors")],
                     layout=go.Layout(
                         title="5-Fold Cross-Validation Errors For Previous Function, "
                         "noise = {}, samples = {}".format(noise, n_samples),
                         xaxis_title="Polynomial Degree",
                         yaxis_title="Average Error",
                         height=400))
    fig2.show()
    fig2.write_image("ex5/errors_noise_{}_samples_{}.png".format(noise, n_samples))

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(np.array(validate_scores))
    best_poly = PolynomialFitting(k_star).fit(train_X, train_y)
    test_error = round(best_poly.loss(test_X.to_numpy().flatten(), test_y.to_numpy()), 2)
    print("Test error for noise {} is {} for k_star = {}".
          format(noise, test_error, k_star))

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions
    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate
    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X = X[:n_samples, :]
    train_y = y[:n_samples]
    test_X = X[n_samples:, :]
    test_y = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_space = np.linspace(0.01, 10, num=n_evaluations)
    ridge_train_scores, ridge_validate_scores = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lasso_train_scores, lasso_validate_scores = np.zeros(n_evaluations), np.zeros(n_evaluations)
    for i, lam in enumerate(lambda_space):
        ridge_train_scores[i], ridge_validate_scores[i] = \
            cross_validate(estimator=RidgeRegression(lam), X=train_X, y=train_y,
                           scoring=mean_square_error)
        lasso_train_scores[i], lasso_validate_scores[i] = \
            cross_validate(estimator=Lasso(alpha=lam), X=train_X, y=train_y,
                           scoring=mean_square_error)
    fig3 = make_subplots(rows=1, cols=2, subplot_titles="{} Estimator".format(["Ridge", "Lasso"]),
                        horizontal_spacing=0.1, vertical_spacing=0.1)
    fig3.add_traces(go.Scatter(x=lambda_space, y=ridge_train_scores, mode='markers',
                    name="Ridge train scores",
                    marker=dict(color="black", opacity=0.7)), rows=1, cols=1)
    fig3.add_traces(go.Scatter(x=lambda_space, y=ridge_validate_scores, mode='markers',
                    name="Ridge validation scores",
                    marker=dict(color="orange", opacity=0.7)), rows=1, cols=1)
    fig3.add_traces(
        go.Scatter(go.Scatter(x=lambda_space, y=lasso_train_scores, mode='markers',
                   name="Lasso train scores",
                   marker=dict(color="black", opacity=0.7))), rows=1, cols=2)
    fig3.add_traces(go.Scatter(x=lambda_space, y=lasso_validate_scores, mode='markers',
                    name="Lasso validation scores",
                    marker=dict(color="orange", opacity=0.7)), rows=1, cols=2)
    fig3.update_layout(title="Train and Validation Errors as a function of Regularization Parameter Value",
                             xaxis_title="Regularization Parameter Value",
                             yaxis_title="Errors",
                             height=400)
    fig3.show()
    fig3.write_image("ex5/ridge_and_lasso_errors.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_idx = np.argmin(ridge_validate_scores)
    ridge_lam_star = lambda_space[ridge_idx]
    best_ridge = RidgeRegression(ridge_lam_star).fit(train_X, train_y)
    ridge_test_errors = best_ridge.loss(test_X, test_y)
    print("Test error for ridge estimator is {} for lambda_star = {}".
          format(ridge_test_errors, ridge_lam_star))

    lasso_idx = np.argmin(lasso_validate_scores)
    lasso_lam_star = lambda_space[lasso_idx]
    best_lasso = Lasso(lasso_lam_star).fit(train_X, train_y)
    lasso_test_errors = round(mean_square_error(test_y, best_lasso.predict(test_X)), 3)
    print("Test error for lasso estimator is {} for lambda_star = {}".
          format(lasso_test_errors, lasso_lam_star))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)  # Question 4
    select_polynomial_degree(n_samples=1500, noise=0)  # Question 5
    select_regularization_parameter()
