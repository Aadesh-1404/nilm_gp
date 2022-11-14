import torch

# TODO: Remove this function once gpytorch version of metrics is published.


def negative_log_predictive_density(dist, y):
    """
    Negative log predictive density of model (normalized by number of observations).
    """
    return -dist.log_prob(y.ravel()).item() / y.shape[0]


def mean_squared_error(dist, y, squared=True):
    """
    Mean Squared Error
    """
    mse = torch.square(y.ravel() - dist.mean).mean().item()
    if not squared:
        return mse**0.5  # Root mean square error
    return mse
