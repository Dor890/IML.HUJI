from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else\
        (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets.

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y_true = load_dataset('../datasets/{}'.format(f))

        # Fit Perceptron and record loss in each fit iteration
        def loss_recorder(fit: Perceptron, x: np.ndarray, y: int):
            """
            Calculates the loss value in every iteration of the Perceptron algorithm,
            according to the updated coefficients vector, and stores it in the
            training_loss_ array.

            Parameters
            ----------
            fit: Perceptron instance

            X : ndarray of shape (n_features, )
                Current sample

            y : int
                Current response
            """
            fit.training_loss_.append(fit._loss(X, y_true))

        y_pred = Perceptron(callback=loss_recorder).fit(X, y_true)
        losses = y_pred.training_loss_

        # Plot figure
        go.Figure([go.Scatter(x=[i for i in range(len(losses))], y=losses, mode='lines',
                              line=dict(width=3, color="rgb(204,68,83)"),
                              name='# misclassification errors')],
                  layout=go.Layout(barmode='overlay',
                                   title="Training Loss Values for {} data".format(n),
                                   xaxis_title="Training Iterations",
                                   yaxis_title="Training Loss Values",
                                   height=500)).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset('../datasets/{}'.format(f))

        # Fit models and predict over training set
        LDA_pred = LDA().fit(X, y)
        GNB_pred = GaussianNaiveBayes().fit(X, y)
        lda_y_pred = LDA_pred.predict(X).astype('int64')
        gnb_y_pred = GNB_pred.predict(X).astype('int64')

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left
        # and LDA predictions on the right.
        # Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        # Create subplots
        model_names = [('Gaussian Naive Bayes', round(accuracy(y, gnb_y_pred), 3)),
                     ('Linear Discriminant Analysis (LDA)', accuracy(y, lda_y_pred))]
        symbols = np.array(['circle', 'square', 'cross'])
        colors = np.array(['red', 'blue', 'green'])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["{} Classifier with Accuracy: {}".format(m[0], m[1])
                                            for m in model_names],
                            horizontal_spacing=0.1, vertical_spacing=0.1)

        # LDA Sub-plot
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=colors[lda_y_pred], symbol=symbols[y]))],
                       rows=1, cols=2)

        # GBA Sub-plot
        fig.add_traces(
            [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=colors[gnb_y_pred], symbol=symbols[y]))],
            rows=1, cols=1)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=LDA_pred.mu_[:, 0], y=LDA_pred.mu_[:, 1],
                                   mode="markers", showlegend=False,
                        marker=dict(color='black', symbol='x'))],
                       rows=1, cols=2)
        fig.add_traces([go.Scatter(x=GNB_pred.mu_[:, 0], y=GNB_pred.mu_[:, 1],
                                   mode="markers", showlegend=False,
                                   marker=dict(color='black', symbol='x'))],
                       rows=1, cols=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_traces(get_ellipse(LDA_pred.mu_[0], LDA_pred.cov_), rows=1, cols=2)
        fig.add_traces(get_ellipse(LDA_pred.mu_[1], LDA_pred.cov_), rows=1, cols=2)
        fig.add_traces(get_ellipse(LDA_pred.mu_[2], LDA_pred.cov_), rows=1, cols=2)
        fig.add_traces(get_ellipse(GNB_pred.mu_[0], np.diag(GNB_pred.vars_[0])), rows=1, cols=1)
        fig.add_traces(get_ellipse(GNB_pred.mu_[1], np.diag(GNB_pred.vars_[1])), rows=1, cols=1)
        fig.add_traces(get_ellipse(GNB_pred.mu_[2], np.diag(GNB_pred.vars_[2])), rows=1, cols=1)

        # Final step - updating titles and showing the plot
        fig.update_layout(title="Results for Predicted Class vs. True Class",
                          xaxis={"title": "Feature 1"},
                          yaxis={"title": "Feature 2"},
                          height=400)
        fig.show()

def quiz_question():
    X1 = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
    y1 = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    X2 = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    y2 = np.array([0, 0, 1, 1, 1, 1])

    GNB = GaussianNaiveBayes()
    GNB.fit(X2, y2)
    print(GNB.vars_)
    print(GNB.mu_)
    # print(GNB.pi_)
    # print(GNB.likelihood(X2))

if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    quiz_question()
