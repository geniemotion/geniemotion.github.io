import warnings
import numpy as np
import torch
import gpytorch

# Predictions are made at the exact training inputs, which makes gpytorch emit
# a warning about the input matching the stored training data; silence it.
import shutup
shutup.please()

class _BatchExactGP(gpytorch.models.ExactGP):
    """One GP per entry of `batch_shape` — e.g. batch_shape=(V, 3) means
    V vertices x 3 output channels, all fit in a single vectorized op."""

    def __init__(self, train_x, train_y, likelihood, batch_shape):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchGPWarper:
    """
    Fits one independent GP per (vertex, output-dim) pair -- V*3 total GPs --
    as a single batched tensor computation.

        batch_gp = BatchGPWarper()
        warped = batch_gp.fit_predict(src_affine_batch, target)

    where src_affine_batch/target are (V, N, 3) arrays (V = n_vertices,
    N = n_frames). Each GP has a constant mean and a scaled RBF kernel and
    models the residual target - source; hyperparameters are optimised with
    Adam (n_iter steps). float64 is used by default.
    """

    def __init__(self, length_scale_init=0.2, noise_init=1e-5,
                 device=None, dtype=torch.float64, lr=0.05, n_iter=500,
                 verbose=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.length_scale_init = length_scale_init
        self.noise_init = noise_init
        self.lr = lr
        self.n_iter = n_iter
        self.verbose = verbose

    def fit_predict(self, source, target):
        """
        source, target: (V, N, 3) numpy arrays.
        Returns warped: (V, N, 3) numpy array = source + predicted residual,
        evaluated at the training points (matches GPWarper.predict(source)).
        """
        source = np.asarray(source)
        target = np.asarray(target)
        V, N, D = source.shape
        assert target.shape == (V, N, D), "source/target shape mismatch"

        residual = target - source  # GP learns the residual field, same as GPWarper

        batch_shape = torch.Size([V, D])  # one GP per (vertex, output channel)

        x = torch.as_tensor(source, dtype=self.dtype, device=self.device)   # (V, N, D)
        # every output channel of a vertex shares the same input locations
        train_x = x.unsqueeze(1).expand(V, D, N, D).contiguous()            # (V, D, N, D)

        y = torch.as_tensor(residual, dtype=self.dtype, device=self.device)  # (V, N, D)
        train_y = y.permute(0, 2, 1).contiguous()                            # (V, D, N)

        # GaussianLikelihood's default noise lower bound (1e-4) is above
        # noise_init (1e-5), so widen it to allow that init.
        noise_constraint = gpytorch.constraints.GreaterThan(
            min(1e-6, self.noise_init * 0.1)
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=batch_shape, noise_constraint=noise_constraint
        ).to(self.device, self.dtype)
        model = _BatchExactGP(train_x, train_y, likelihood, batch_shape).to(
            self.device, self.dtype
        )

        model.covar_module.base_kernel.lengthscale = self.length_scale_init
        likelihood.noise = self.noise_init

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.n_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).sum()  # sum over the (V, D) batch
            loss.backward()
            optimizer.step()
            if self.verbose and i % 25 == 0:
                print(f"[BatchGPWarper] iter {i:4d}  loss {loss.item():.4f}")

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(train_x)).mean  # (V, D, N) predicted residual

        pred_residual = pred.permute(0, 2, 1).cpu().numpy()  # (V, N, D)
        return source + pred_residual