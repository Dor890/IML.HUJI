from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    MU, VAR = 10, 1
    samples = np.random.normal(MU, VAR, size=1000)
    # print("Univariate samples are:")
    # print(samples)
    uni_gaussian = UnivariateGaussian()
    uni_gaussian.fit(samples)
    print("Estimated (expectation, variance):")
    print((uni_gaussian.mu_, uni_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int64)
    estimated_diff = []
    for m in ms:
        uni_gaussian.fit(samples[:m])
        estimated_diff.append(np.abs(uni_gaussian.mu_ - MU))

    go.Figure([go.Scatter(x=ms, y=estimated_diff, mode='markers+lines',
                          name=r'r$|\hat\mu - \mu|$'),
               go.Scatter(x=ms, y=[0]*len(ms), mode='lines',
                          name=r'0 - Expected Value')],
              layout=go.Layout(
                  title=r"$\text{|Estimated - True Val. of Expectation| As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    theoretical_dist_m = uni_gaussian.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=theoretical_dist_m, mode='markers',
                          line=dict(width=3, color="rgb(204,68,83)"),
                          name=r'$N(\hat\mu, \hat\sigma)$')],
              layout=go.Layout(barmode='overlay',
                               title="The Empirical PDF Function Under The Fitted Model",
                               xaxis_title="Ordered Sample Values",
                               yaxis_title="Probability density function - $PDF$",
                               height=500)).show()

    # Quiz calculations
    # samples2 = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1,
    #                      1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #                      -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1,
    #                      4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # print('Log-likelihood')
    # print(uni_gaussian.log_likelihood(10, 1, samples2))



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, sigma, size=1000)
    # print("Multivariate samples are:")
    # print(samples)
    multi_gaussian = MultivariateGaussian()
    multi_gaussian.fit(samples)
    print("Estimated expectation vector:")
    print(multi_gaussian.mu_)
    print("Estimated covariance matrix:")
    print(multi_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f_values = np.linspace(-10, 10, 200)
    log_likelihood_vals = []
    for f3 in f_values:
        f3_vals = []
        for f1 in f_values:
            f3_vals.append(
                multi_gaussian.log_likelihood(
                    np.array([f1, 0, f3, 0]), sigma, samples))
        log_likelihood_vals.append(f3_vals)
    log_likelihood_vals = np.array(log_likelihood_vals)
    go.Figure(data=go.Heatmap(x=f_values, y=f_values, z=log_likelihood_vals),
        layout=go.Layout(xaxis_title="$f1$", yaxis_title="$f3$",
        title=r"$\text{Log-Likelihood With Scaling Mean}$"
    )).show()

    # Question 6 - Maximum likelihood
    max_index = np.argmax(log_likelihood_vals)
    max_indices = np.unravel_index(max_index, shape=log_likelihood_vals.shape)
    print("maxarg for f1 is {} and for f3 is {}".format(
        f_values[max_indices[1]], f_values[max_indices[0]]))

    # Quiz calculations
    # print('Cov of 1 and 4')
    # print(multi_gaussian.cov_[0][3])
    # print("CHECK")
    # print(multi_gaussian.pdf(samples))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
