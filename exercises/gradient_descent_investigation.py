import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
from sklearn.metrics import roc_curve

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm
    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted
    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path
    title: str, default=""
        Setting details to add to plot title
    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown
    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration
    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm
    values: List[np.ndarray]
        Recorded objective values
    weights: List[np.ndarray]
        Recorded parameters
    """
    values_arr, weights_arr = [], []
    def fresh_callback(**kwargs):
        values_arr.append(kwargs['val'])
        weights_arr.append(kwargs['weights'])
    return fresh_callback, values_arr, weights_arr


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [[L1, 'L1 module'], [L2, 'L2 module']]
    for module in modules:
        min_loss = 1
        cov_fig = go.Figure(layout=go.Layout(
            title="Convergence Rate of {}".format(module[1]),
            xaxis_title="Norm",
            yaxis_title="GD Iteration",
            height=400))
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            lr = FixedLR(eta)
            cur_module = module[0](init)
            gd_algorithm = GradientDescent(learning_rate=lr, callback=callback)
            gd_algorithm.fit(cur_module, np.zeros(1), np.zeros(1))
            path_fig = plot_descent_path(module[0], np.array(weights),
                              title='of {} with η = {}'.format(module[1], eta))
            path_fig.write_image("ex6/GD Descent Path of {} with eta {}.png".format(module[1], eta))
            # path_fig.show()
            if values[-1] < min_loss:
                min_loss = values[-1]
            cov_fig.add_traces([go.Scatter(
                x=[i for i in range(len(values))],
                y=values, mode='markers', name='Eta = {}'.format(eta))])
        cov_fig.write_image("ex6/Convergence Rate of {} with Fixed lr.png".format(module[1]))
        # cov_fig.show()
        print('Lowest Loss Achieved When Minimizing {} is {}'.format(module[1], min_loss))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    cov_fig = go.Figure(layout=go.Layout(
        title="Convergence Rate of L1 module with Exponential Decay",
        xaxis_title="Norm",
        yaxis_title="GD Iteration",
        height=400))
    min_loss = 1
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        lr = ExponentialLR(eta, gamma)
        gd_algorithm = GradientDescent(learning_rate=lr, callback=callback)
        l1_model = L1(init)
        gd_algorithm.fit(l1_model, np.zeros(1), np.zeros(1))
        cov_fig.add_traces([go.Scatter(
            x=[i for i in range(len(values))],
            y=values, mode='lines', name='Gamma = {}'.format(gamma))])
        if values[-1] < min_loss:
            min_loss = values[-1]
        if gamma == 0.95:
            l1_fig = plot_descent_path(L1, np.array(weights),
                                       title='of L1 module with gamma = 0.95')
            callback, values, weights = get_gd_state_recorder_callback()
            gd_algorithm = GradientDescent(learning_rate=lr, callback=callback)
            l2_model = L2(init)
            gd_algorithm.fit(l2_model, np.zeros(1), np.zeros(1))
            l2_fig = plot_descent_path(L2, np.array(weights),
                                       title='of L2 module with gamma = 0.95')
    print('Lowest l1 Norm Achieved Using Exponential Decay is {}'.format(min_loss))

    # Plot algorithm's convergence for the different values of gamma
    cov_fig.write_image("ex6/Convergence Rate of L1 module with Exponential Decay.png")
    # cov_fig.show()

    # Plot descent path for gamma=0.95
    l1_fig.write_image("ex6/GD Descent Path of L1 module with Exponential Decay.png")
    # l1_fig.show()
    l2_fig.write_image("ex6/GD Descent Path of L2 module with Exponential Decay.png")
    # l2_fig.show()

def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion
    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset
    train_portion: float, default=0.8
        Portion of dataset to use as a training set
    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set
    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples
    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set
    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(),\
                                       X_test.to_numpy(), y_test.to_numpy()
    logistic = LogisticRegression().fit(X_train, y_train)

    # Plotting convergence rate of logistic regression over SA heart disease data
    y_proba = logistic.predict_proba(X_train)
    # fpr, tpr, thresholds = roc_curve(y_train, y_proba)
    alpha_space = np.linspace(0, 1, 101)
    fpr, tpr = np.ndarray(alpha_space.size), np.ndarray(alpha_space.size)
    P, N = np.count_nonzero(y_train == 1), np.count_nonzero(y_train == 0)
    for i, alpha in enumerate(alpha_space):
        fp = np.sum(np.logical_and(y_proba > alpha, y_train == 0))
        fpr[i] = fp / N
        tp = np.sum(np.logical_and(y_proba > alpha, y_train == 1))
        tpr[i] = tp / P
    values = tpr - fpr
    best_idx = np.argmax(values)
    alpha_star = alpha_space[best_idx]
    print('Best alpha found in ROC is {}'.format(alpha_star))

    roc_fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    line=dict(color="black", dash='dash'),
                                    name="Random Class Assignment"),
                         go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=alpha_space,
                                    name="", showlegend=False, marker_color='blue',
                                    hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                        layout=go.Layout(title='ROC Curve of Logistic Regression over dataset',
                                         xaxis_title='False Positive Rate (FPR)',
                                         yaxis_title='True Positive Rate (TPR)'))
    roc_fig.write_image('ex6/Logistic ROC2.png')
    # roc_fig.show()
    best_logistic = LogisticRegression(alpha=alpha_star).fit(X_train, y_train)
    print('Model’s test error with best alpha is {}'.format(best_logistic.loss(X_test, y_test)))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambda_space = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    train_scores_l1, validate_scores_l1, train_scores_l2, validate_scores_l2 =\
        np.ones(len(lambda_space)), np.ones(len(lambda_space)), \
        np.ones(len(lambda_space)), np.ones(len(lambda_space))

    for i, lam in enumerate(lambda_space):
        train_scores_l1[i], validate_scores_l1[i] = cross_validate(
            estimator=LogisticRegression(penalty='l1', lam=lam),
            X=X_train, y=y_train, scoring=misclassification_error)
        train_scores_l2[i], validate_scores_l2[i] = cross_validate(
            estimator=LogisticRegression(penalty='l2', lam=lam),
            X=X_train, y=y_train, scoring=misclassification_error)

    # L1
    best_idx_l1 = np.argmin(validate_scores_l1)
    lam_star_l1 = lambda_space[best_idx_l1]
    print('Best lambda for L1 found by CV is {}'.format(lam_star_l1))
    reg_logistic = LogisticRegression(penalty='l1', lam=lam_star_l1).fit(X_train, y_train)
    print('Model’s test error with L1 is {}'.format(reg_logistic.loss(X_test, y_test)))

    # L2
    best_idx_l2 = np.argmin(validate_scores_l2)
    lam_star_l2 = lambda_space[best_idx_l2]
    print('Best lambda for L2 found by CV is {}'.format(lam_star_l2))
    reg_logistic = LogisticRegression(penalty='l2', lam=lam_star_l2).fit(X_train, y_train)
    print('Model’s test error with L2 is {}'.format(reg_logistic.loss(X_test, y_test)))

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
