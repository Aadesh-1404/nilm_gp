import torch
import gpytorch
import warnings
import faiss
import numpy as np
from memory_profiler import profile
from tqdm import tqdm


class BaseRegressor(torch.nn.Module):
    def __init__(self, train_x, train_y, mll):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.mll = mll

        self.history = {}
        self.best_restart = None

    def compute_nn_idx(self, x, k):
        # TODO: I'll utilize this for 60000/40000 problem
        self.cpu_index = faiss.IndexFlatL2(x.size(-1))
        x = np.ascontiguousarray((x.data.float()).cpu().numpy())
        self.cpu_index.add(x)
        return torch.from_numpy(self.cpu_index.search(x, k)[1]).long()
    # @profile

    def loss_func(self, X, y):
        if not self.mll.__class__.__name__ == 'VariationalELBO':
            self.mll.model.set_train_data(X, y, strict=False)

        output = self.mll.model(X)
        return -self.mll(output, y)
    # @profile

    def fit(
        self,
        batch_size=None,
        lr=0.1,
        n_epochs=1,
        n_restarts=1,
        verbose=0,
        verbose_gap=1,
        random_state=None,
        thetas=None,
        x_test=None,
        y_test=None,
    ):
        if random_state is not None:
            torch.manual_seed(random_state)
        if batch_size is None:
            batch_size = self.train_x.shape[0]

        assert batch_size > 0 and batch_size <= self.train_x.shape[0]
        batch_mode = False if batch_size == self.train_x.shape[0] else True

        self.optimizer = torch.optim.Adam(self.mll.parameters(), lr=lr)

        # if batch_mode:
        # train_nn_idx = self.compute_nn_idx(
        #     x=self.train_x, k=batch_size)

        X_batch = self.train_x
        y_batch = self.train_y
        best_loss = float("inf")
        n_iters = max(1, self.train_x.shape[0] // batch_size)
        best_mll_state = None
        self.history["epoch_loss"] = []
        self.history["iter_loss"] = []
        self.history["test_iter_loss"] = []

        for restart in range(n_restarts):
            self.history["epoch_loss"].append([])
            self.history["iter_loss"].append([])
            self.history["test_iter_loss"].append([])
            if restart > 0:  # Don't reset the model if it's the first restart
                for param in self.mll.parameters():
                    torch.nn.init.normal_(param, mean=0.0, std=1.0)
            # self.mll.model.initialize(**thetas)
            if thetas != None:
                self.mll.model.initialize(**thetas)
            for epoch in range(1, n_epochs+1):
                loss = 0
                for iteration in range(1, n_iters+1):
                    self.mll.train()
                    if batch_mode:
                        idx = torch.randint(
                            low=0, high=self.train_x.shape[0], size=(1,)
                        )[0]
                        indices = np.random.choice(
                            range(self.train_x.shape[0]), batch_size, replace=False)
                        X_batch = self.train_x[indices]
                        y_batch = self.train_y[indices]

                    self.optimizer.zero_grad()
                    batch_loss = self.loss_func(X_batch, y_batch)
                    # print(batch_loss)
                    self.history["iter_loss"][restart].append(
                        batch_loss.item())
                    batch_loss.backward()
                    loss += batch_loss.item()
                    if verbose and (iteration % verbose_gap) == 0:
                        print(
                            "restart: {}, epoch: {}, iter: {}, loss: {:.4f}".format(
                                restart, epoch, iteration, loss
                            ), flush=True
                        )
                    self.optimizer.step()
                    if (x_test is not None):
                        pred_dist = self.predict(x_test)
                        # print(pred_dist.mean.dtype, y_test.dtype)
                        # print(-pred_dist.log_prob(y_test))
                        test_loss = -pred_dist.log_prob(y_test)/x_test.numel()
                        self.history["test_iter_loss"][restart].append(
                            test_loss)

                loss = loss / n_iters
                self.history["epoch_loss"][restart].append(loss)

            # Check if best loss
            if loss < best_loss:
                self.best_restart = restart
                best_loss = loss
                best_mll_state = self.mll.state_dict()

        # Load the best model
        if best_mll_state is not None:
            self.mll.load_state_dict(best_mll_state)

        return self.history["iter_loss"]

    def predict(self, X_test):
        if not self.mll.__class__.__name__ == 'VariationalELBO':
            self.mll.model.set_train_data(
                self.train_x, self.train_y, strict=False)

        self.mll.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.pred_dist = self.mll.likelihood(self.mll.model(X_test))
            return self.pred_dist

    def predict_batch(self, X_test, batch_size, nn_size):
        # TODO: Work in progress here
        if self.mll.__class__.__name__ == 'VariationalELBO':
            raise NotImplementedError(
                'Batch prediction not implemented for VariationalELBO')
        self.mll.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # kmeans = faiss.Kmeans(X_test.shape[1], pred_train_len, niter=1024)
            # kmeans.train(x)
            # centroids = kmeans.centroids

            # kmeans = KMeans(n_clusters=pred_train_len, random_state=0).fit(x)
            # centroids = kmeans.cluster_centers_

            x = (X_test.data.float()).cpu().numpy()
            faiss_index = faiss.IndexFlatL2(self.train_x.size(-1))
            if self.train_x.device == 'cuda':
                res = faiss.StandardGpuResources()
                faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
                faiss_index.add(np.ascontiguousarray(x))
            else:
                faiss_index.add(np.ascontiguousarray(x))

            x_batches = X_test.split(batch_size)
            means = torch.zeros(len(X_test)).to(self.train_x.device)
            variances = torch.zeros(len(X_test)).to(self.train_x.device)
            for i, x_batch in tqdm(enumerate(x_batches)):
                centroids = np.ascontiguousarray(
                    (x_batch.data.float()).cpu().numpy())
                train_nn_idx = torch.from_numpy(
                    faiss_index.search(centroids, nn_size)[1]).long()
                train_nn_idx = train_nn_idx.reshape(-1).unique()
                self.mll.model.set_train_data(
                    self.train_x[train_nn_idx], self.train_y[train_nn_idx], strict=False)
                pred_dist = self.mll.likelihood(self.mll.model(x_batch))
                means[i*batch_size:(i+1)*batch_size] = pred_dist.mean
                variances[i*batch_size:(i+1)*batch_size] = pred_dist.variance
            self.mll.model.set_train_data(
                self.train_x, self.train_y, strict=False)

        return means, variances
