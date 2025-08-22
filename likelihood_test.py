# Test monotonic likelihood function defined in rm_hooks.py
import numpy as np
import jax.numpy as jnp
import jax.random as random

from numpyro import distributions as dist
from numpyro import deterministic, sample, factor
from numpyro.infer import MCMC, NUTS, init_to_median

import matplotlib.pyplot as plt
import corner

from rm_hooks import global_monotonic_ll

def test_function(x, a:float, b:float, c:float, d:float):
    return a * jnp.cos(x) + b * x + c * jnp.exp(x) + d

def ll_test(x):
    a = sample("a", dist.TruncatedNormal(0.0, 1.0, low=-3.0, high=3.0))
    b = sample("b", dist.TruncatedNormal(0.0, 1.0, low=-3.0, high=3.0))
    c = sample("c", dist.TruncatedNormal(0.0, 1.0, low=-3.0, high=3.0))
    d = sample("d", dist.TruncatedNormal(0.0, 1.0, low=-3.0, high=3.0))
    y = deterministic("y", test_function(x, a, b, c, d))
    ll = global_monotonic_ll(y, jnp.ones_like(y)*0.1)
    deterministic("log_likelihood", ll)
    factor("ll", ll)

x = jnp.linspace(-5, 5, 100)
nuts_kernel = NUTS(ll_test, init_strategy=init_to_median(num_samples=1000))
mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=20000, progress_bar=True)
mcmc.run(random.PRNGKey(0), x)
mcmc.print_summary()
samples = mcmc.get_samples()

max_ll_idx = np.argmax(samples["log_likelihood"])
print("Max likelihood parameters: ")
print(f"a = {samples['a'][max_ll_idx]:.2f}")
print(f"b = {samples['b'][max_ll_idx]:.2f}")
print(f"c = {samples['c'][max_ll_idx]:.2f}")
print(f"d = {samples['d'][max_ll_idx]:.2f}")

corner_samples = {
    "a": samples["a"],
    "b": samples["b"],
    "c": samples["c"],
    "d": samples["d"]
}
corner_data = np.column_stack([corner_samples[key] for key in ["a", "b", "c", "d"]])
figure = corner.corner(
    corner_data,
    labels=["a", "b", "c", "d"],
    truths=[
        samples["a"][max_ll_idx],
        samples["b"][max_ll_idx],
        samples["c"][max_ll_idx],
        samples["d"][max_ll_idx]
    ]
)
plt.savefig("/mnt/users/koe/SPARC_RAR/likelihood_test.png", dpi=300)
