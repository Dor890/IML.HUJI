from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
from sklearn.naive_bayes import GaussianNB

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
    return data[:, :2], data[:, 2]


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
            fit.training_loss_.append(fit.loss(X, y_true))

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
        y = y.astype('int64')

        # Fit models and predict over training set
        LDA_pred = LDA().fit(X, y)
        GNB_pred = GaussianNaiveBayes().fit(X, y)
        # GNB_pred = GaussianNB()
        # GNB_pred.fit(X, y)
        lda_y_pred = LDA_pred.predict(X).astype('int64')
        gnb_y_pred = GNB_pred.predict(X).astype('int64')

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left
        # and LDA predictions on the right.
        # Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        # Predicted vs True
        model_names = [('Linear Discriminant Analysis (LDA)', accuracy(y, lda_y_pred)),
                       ('Gaussian Naive Bayes', round(accuracy(y, gnb_y_pred), 3))]
        symbols = np.array(['circle', 'square', 'cross'])
        colors = np.array(['red', 'blue', 'green'])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["{} Classifier with Accuracy: {}".format(m[0], m[1])
                                            for m in model_names],
                            horizontal_spacing=0.1, vertical_spacing=.03)

        # LDA Sub-plot
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=colors[lda_y_pred], symbol=symbols[y]))],
                       rows=1, cols=1)

        # LDA - Adding X's as center of fitted Gaussian
        fig.add_traces([go.Scatter(x=LDA_pred.mu_[:, 0], y=LDA_pred.mu_[:, 1], mode="markers", showlegend=False,
                        marker=dict(color='black', symbol='x'))],
            rows=1, cols=1)

        # LDA - Adding ellipse
        v = np.linalg.eigh(LDA_pred.cov_)[0]
        a = v[1]
        b = v[0]
        axis_sum = np.sum(LDA_pred.mu_, axis=0)
        x_origin = axis_sum[0] / 3
        y_origin = axis_sum[1] / 3
        x_ = []
        y_ = []

        for t in range(0, 361, 10):
            x = a * (np.cos(np.radians(t))) + x_origin
            x_.append(x)
            y__ = b * (np.sin(np.radians(t))) + y_origin
            y_.append(y__)

        fig.add_traces([go.Scatter(x=np.array(x_), y=np.array(y_), mode='lines', showlegend=False,
                        line=dict(color='black', width=2))],
                       rows=1, cols=1)

        # GBA Sub-plot
        fig.add_traces(
            [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=colors[gnb_y_pred], symbol=symbols[y]))],
            rows=1, cols=2)

        # GBA - Adding X's as center of fitted Gaussian
        fig.add_traces([go.Scatter(x=GNB_pred.mu_[:, 0], y=GNB_pred.mu_[:, 1],
                                   mode="markers", showlegend=False,
                                   marker=dict(color='black', symbol='x'))],
                       rows=1, cols=2)

        # GBA - Adding ellipse
        v = np.linalg.eigh(np.diag(np.mean(GNB_pred.vars_, axis=0)))[0]
        a = v[1]
        b = v[0]
        axis_sum = np.sum(GNB_pred.mu_, axis=0)
        x_origin = axis_sum[0] / 3
        y_origin = axis_sum[1] / 3
        x_ = []
        y_ = []

        for t in range(0, 361, 10):
            x = a * (np.cos(np.radians(t))) + x_origin
            x_.append(x)
            y__ = b * (np.sin(np.radians(t))) + y_origin
            y_.append(y__)

        fig.add_traces([go.Scatter(x=np.array(x_), y=np.array(y_), mode='lines', showlegend=False,
                        line=dict(color='black', width=2))],
                       rows=1, cols=2)

        # Final step - updating titles and showing the plot
        fig.update_layout(title="Results for Predicted Class vs. True Class",
                          xaxis={"title": "Feature 1"},
                          yaxis={"title": "Feature 2"},
                          height=400)
        fig.show()

def quiz_question():
    X1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y1 = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    LDA_pred = LDA()
    # LDA_pred.fit(X1, y1)
    # LDA_pred.predict(X1)
    # LDA_pred.likelihood(X1)

    X2 = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    X3 = np.array([[1, 1, 2], [1, 2, 2], [2, 3, 3], [2, 4, 4], [3, 3, 3], [3, 4, 4]])
    y2 = np.array([0, 0, 1, 1, 1, 1])
    # LDA_pred.fit(X2, y2)
    # print(LDA_pred.predict(X2))
    # x = LDA_pred.likelihood(X2)
    # print(LDA_pred.likelihood(X2))
    # for i in x:
    #     print(i[0] > i[1])

    GNB = GaussianNaiveBayes()
    GNB.fit(X2, y2)
    print(GNB.vars_)
    print(GNB.mu_)
    print(GNB.predict(X2))


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
    # quiz_question()
