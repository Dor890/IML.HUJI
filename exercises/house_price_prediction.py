import os
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Passing only over the Non-categorical features which we can measure
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'waterfront', 'view', 'condition',
                'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                'yr_renovated', 'sqft_living15', 'sqft_lot15']
    for feature_name in features:
        feature = X[feature_name]
        pearson_corr = (np.cov(feature, y) / (np.std(feature) * np.std(y)))[0][1]
        fig = go.Figure([go.Scatter(x=feature, y=y, mode='markers',
                                    name="Price according to {}".format(feature_name),
                                    marker=dict(color="blue", opacity=.7))],
                        layout=go.Layout(
                            title="{} Feature With p = {}".format(feature_name,
                                                                  round(pearson_corr, 3)),
                            xaxis_title='{}'.format(feature_name),
                            yaxis_title="response",
                            height=400))
        fig.show()
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        fig.write_image("{}/{}.png".format(output_path, feature_name))


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X = load_data("..\datasets\house_prices.csv")
    # Dropping unnecessary features
    X.drop(columns=['id', 'date', 'lat', 'long'], inplace=True)
    # Removing Non-positive samples
    X = X[X.select_dtypes(include=[np.number]).ge(0).all(1)]
    # Removing missing values
    X.dropna(axis=0, how='any')
    # Handling categorical features
    X = pd.get_dummies(X, columns=['zipcode'])
    # Separating price column
    y = X['price']
    X.drop('price', inplace=True, axis=1)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, './house_price_plots')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
