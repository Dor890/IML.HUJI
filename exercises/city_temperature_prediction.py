import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = pd.read_csv('..\datasets\City_Temperature.csv', parse_dates=['Date'])
    df.dropna(axis=0, how='any', inplace=True)
    df = df[df.select_dtypes(include=[np.number]).ge(-20).all(1)]
    df['DayOfYear'] = [date.day_of_year for date in df['Date']]

    # Question 2 - Exploring data for specific country
    df["Year"] = df["Year"].astype(str)
    il_data = df.loc[df['Country'] == 'Israel']
    fig = px.scatter(il_data, x=il_data['DayOfYear'], y=il_data['Temp'],
                     color='Year',
                     title="Average Daily Temperature as function of DayOfYear")
    fig.show()

    standard_dev = il_data.groupby('Month').agg('std')['Temp']
    fig = px.bar(x=range(1, 13), y=standard_dev,
                 title="Standard Deviation Of The Daily Temperatures by Months")
    fig.update_layout(xaxis_title="Month", yaxis_title="Temp Deviant")
    fig.show()

    # Question 3 - Exploring differences between countries
    grouped_multiple = df.groupby(['Country', 'Month']).agg(['mean', 'std'])[
        'Temp']
    mean_vals = grouped_multiple['mean'].to_numpy()
    std_vals = grouped_multiple['std'].to_numpy()
    df["Country"] = df["Country"].astype(str)
    x_vals = [i for i in range(1, 13)] * df['Country'].nunique()
    fig = px.line(df, x=x_vals, y=mean_vals,
                  title='Average Monthly Temperature with Deviation Error Bars, By Countries',
                  error_y=std_vals)
    fig.update_layout(xaxis_title="Month", yaxis_title="Average Temp")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    X = il_data['DayOfYear']
    y = il_data['Temp']
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    loss_arr = []
    for k in range(11):
        poly_est = PolynomialFitting(k).fit(train_X.to_numpy(), train_y.to_numpy())
        cur_loss = round(poly_est.loss(test_X.to_numpy(), test_y.to_numpy()), 2)
        loss_arr.append(cur_loss)
        print(cur_loss)
    fig = px.bar(x=range(11), y=loss_arr,
                 title="Test Error Recorded For Each Value Of k")
    fig.update_layout(xaxis_title="k - Polynomial deg.", yaxis_title="test error")
    fig.show()

# Question 5 - Evaluating fitted model on different countries
    k = 3
    poly_est = PolynomialFitting(k).fit(X.to_numpy(), y.to_numpy())
    loss_arr = []
    countries = df.Country.unique()
    countries = np.delete(countries, np.where(countries == 'Israel')[0][0])
    for country in countries:
        data = df.loc[df['Country'] == country]
        test_X = data['DayOfYear']
        test_y = data['Temp']
        loss_arr.append(round(poly_est.loss(test_X.to_numpy(), test_y.to_numpy()), 2))
    fig = px.bar(x=countries, y=loss_arr,
                 title="Test Error Recorded For Each Country")
    fig.update_layout(xaxis_title="Country",
                      yaxis_title="test error")
    fig.show()
