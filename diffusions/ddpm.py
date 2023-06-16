import abc
import jax
import jax.numpy as jnp
import numpy as np

def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)

from .basic_sde import SDE

class DDPM(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = jnp.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)

        self.alphas = 1. - self.discrete_betas
        self.sqrt_1m_alphas_cumprod = self.sqrt_betas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
        self.discrete_lambda = jnp.log(self.sqrt_alphas_cumprod / self.sqrt_betas_cumprod + 1e-8)
        self.t_array = jnp.linspace(0., 1., N + 1)[1:].reshape((1, -1))

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1 - jnp.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.
        return jax.vmap(logp_fn)(z)

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        sqrt_beta = jnp.sqrt(beta)
        f = batch_mul(jnp.sqrt(alpha), x) - x
        G = sqrt_beta
        return f, G
