import gpytorch
from .base import BaseRegressor


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, kernel, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPRegressor(BaseRegressor):
    """[summary]
    Call the constructor of base class after defining the model.
    """

    def __init__(self, train_x, train_y, kernel, inducing_points):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SVGPModel(kernel, inducing_points)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        super().__init__(train_x, train_y, mll)
