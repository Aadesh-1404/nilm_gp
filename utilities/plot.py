# Refernce: https://github.com/VibhutiBansal-11/NILM_Uncertainty

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from probml_utils import latexify, savefig, is_latexify_enabled
import os
import scipy.stats as st
import jax.numpy as jnp

# os.chdir("../")s

os.environ["LATEXIFY"] = "1"
os.environ["FIG_DIR"] = "./Figures/"


def prediction_plots(x, y_test, y_mean, start, idx, var_pred, fig_name, i):

    latexify(width_scale_factor=2, fig_height=1.75)
    df = pd.read_csv("./time_stamp.csv", index_col=0)
    df.index = df["0"]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Time"
    df = df.drop(columns=["0"])
    df["Main Power"] = x
    df["Ground Truth"] = y_test.cpu()
    df["Prediction"] = y_mean
    df[start : start + idx].plot(rot=90, legend=False)
    plt.fill_between(
        df.index[start : start + idx],
        y_mean[start : start + idx].flatten()
        - 1.96 * np.sqrt(var_pred[start : start + idx]).flatten(),
        y_mean[start : start + idx].flatten()
        + 1.96 * np.sqrt(var_pred[start : start + idx]).flatten(),
        color="lightblue",
        alpha=1.0,
        label="CI (95\%)",
    )
    sns.despine()
    if i == 1:
        plt.legend(frameon=False, bbox_to_anchor=(0.55, 0.45), prop={"size": 7})
    plt.ylabel("Power")

    savefig(fig_name)


def calibration_regression(mean, sigma, Y, label, color, ax=None):
    """
    mean : (n_samples,1) or (n_sample,) prediction mean
    sigma : (n_samples,1) or (n_sample,) prediction sigma
    Y : (n_samples,1) or (n_sample,) Y co-ordinate of ground truth
    label :  string,


    """

    marker_size = 6 if is_latexify_enabled else None
    if ax is None:
        fig, ax = plt.subplots()
    df = pd.DataFrame()
    df["mean"] = mean
    df["sigma"] = sigma
    df["Y"] = Y
    df["z"] = (df["Y"] - df["mean"]) / df["sigma"]
    df["perc"] = st.norm.cdf(df["z"])
    k = jnp.arange(0, 1.1, 0.1)
    counts = []
    df2 = pd.DataFrame()
    df2["Interval"] = k
    df2["Ideal"] = k
    for i in range(0, 11):
        l = df[df["perc"] < 0.5 + i * 0.05]
        l = l[l["perc"] >= 0.5 - i * 0.05]
        counts.append(len(l) / len(df))
    df2["Counts"] = counts

    ax.plot(k, counts, color=color, label=label)

    ax.scatter(k, counts, color=color, s=marker_size)
    ax.scatter(k, k, color="green", s=marker_size)
    ax.set_yticks(k)
    ax.set_xticks(k)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.legend()
    ax.set_xlabel("decile")
    ax.set_ylabel("ratio of points")
    ax.plot(k, k, color="green")
    sns.despine()
    return df, df2
