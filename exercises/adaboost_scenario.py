import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_ensemble = AdaBoost(iterations=n_learners, wl=DecisionStump).fit(train_X, train_y)
    test_errors_arr = [adaboost_ensemble.partial_loss(test_X, test_y, i) for i in range(1, n_learners+1)]

    q1_plot = go.Figure([go.Scatter(x=[i for i in range(n_learners)],
                          y=[adaboost_ensemble.partial_loss(train_X, train_y, i) for i in range(1, n_learners+1)],
                          mode='lines', line=dict(width=3, color="blue"),
                          showlegend=True, name="Train Data"),
                        go.Scatter(x=[i for i in range(n_learners)],
                                   y=test_errors_arr,
                                   mode='lines', line=dict(width=3, color="red"),
                                   showlegend=True, name="Test Data")],
                        layout=go.Layout(barmode='overlay',
                        title="Number of Errors as a Function of The Number of Fitted Learners - Noise = {}".format(noise),
                        xaxis_title="Fitted Learners",
                        yaxis_title="Errors",
                        height=500))
    q1_plot.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(['circle', 'square'])
    q2_plot = make_subplots(rows=2, cols=2,
                            subplot_titles=["{} Iterations".format(m) for m in T],
                            horizontal_spacing=0.12, vertical_spacing=0.3)
    for i, m in enumerate(T):
        q2_plot.add_traces([decision_surface(lambda X : adaboost_ensemble.partial_predict(T=m, X=X), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode="markers", showlegend=False,
                       marker=dict(color=test_y,
                              colorscale=[custom[0], custom[-1]],
                              line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    q2_plot.update_layout(title="Decision Boundary with Increasing Number of Iterations - with Noise={}".format(noise),
                          height=400, margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    q2_plot.show()

    # Question 3: Decision surface of best performing ensemble
    size_for_min = np.argmin(test_errors_arr)
    q3_plot = go.Figure([decision_surface(
        lambda X: adaboost_ensemble.partial_predict(X=X, T=size_for_min),
        lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                   line=dict(color="black", width=1)))],
        layout=go.Layout(title="Ensemble with Lowest Test Error Is On Size {} with Accuracy {} - Noise={}".format(size_for_min+1, 1-test_errors_arr[size_for_min], noise)))
    q3_plot.show()
    q3_plot.write_image("ex4/q3_noise_{}.png".format(noise))

    # Question 4: Decision surface with weighted samples
    norm_weights = adaboost_ensemble.D_ / np.max(adaboost_ensemble.D_) * 5
    q4_plot = go.Figure([decision_surface(adaboost_ensemble.predict, lims[0], lims[1], showscale=False),
                         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                         showlegend=False, marker=dict(color=train_y, size=norm_weights,
                         colorscale=[custom[0], custom[-1]],
                         line=dict(color="black", width=1)))],
                         layout=go.Layout(title="Training Set with Last Iter. Weights - Noise={}".format(noise)))
    q4_plot.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
