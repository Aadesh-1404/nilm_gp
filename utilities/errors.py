import jax.numpy as jnp
import jax
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import torch
from math import pi


def mae(y_mean, y_test):
    return torch.abs(y_mean - y_test).mean(dim=-1)


def msll(var_pred, y_mean, y_test):
    f_var = torch.tensor(var_pred)
    f_mean = torch.tensor(y_mean)
    return 0.5 * (
        torch.log(2 * pi * f_var) + torch.square(y_test - f_mean) / (2 * f_var)
    ).mean(dim=-1)


def qce(std_pred, y_mean, y_test):
    quantile = 95.0
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = torch.tensor(y_mean) - deviation * std_pred
    upper = torch.tensor(y_mean) + deviation * std_pred
    n_samples_within_bounds = ((y_test > lower) * (y_test < upper)).sum(-1)
    fraction = n_samples_within_bounds / y_test.shape[-1]
    return torch.abs(fraction - quantile / 100)


def ace(ideal, predicted):
    """
    dataframe : pandas dataframe with Ideal and Counts as column for regression calibration
    It can be directly used as 2nd output from calibration_regression in plot.py
    """

    def rmse_loss(y, yhat):
        return jnp.abs(y - yhat)

    return jnp.mean(jax.vmap(rmse_loss, in_axes=(0, 0))(ideal, predicted))


def mass_to_std_factor(mass=0.95):
    rv = norm(0.0, 1.0)
    std_factor = rv.ppf((1.0 + mass) / 2)
    return std_factor


def plot_find_p(y, mean_prediction, std_prediction, mass=0.95):
    std_factor = mass_to_std_factor(mass)
    idx = np.where(
        (y < mean_prediction + std_factor * std_prediction)
        & (y > mean_prediction - std_factor * std_prediction)
    )[0]

    p_hat = len(idx) / len(y)
    return (mass, p_hat)


def find_p_hat(y, mean_prediction, std_prediction):
    out = {}
    for mass in np.linspace(1e-10, 1 - 1e-20, 1000):
        out[mass] = plot_find_p(y, mean_prediction, std_prediction, mass)[1]
    df = pd.Series(out).to_frame()
    df.index.name = "p"
    df.columns = ["p_hat"]

    return df
