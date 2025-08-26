#!/usr/bin/env python
# Generate mock hooked (but "unhookable") data to test lieklihood functions in rm_hooks.py.
from jax import numpy as jnp
import jax.random as random
import argparse

import numpyro
from numpyro import distributions as dist
from numpyro import deterministic, sample, factor
from numpyro.infer import MCMC, NUTS, init_to_median

import numpy as np
import corner
import matplotlib.pyplot as plt

from rm_hooks import global_monotonic_ll, discrete_monotonic_ll

def hook_function_1d(a:float=2.0, b:float=-3.0, c:float=-1.0):
    """
    Generates a simple downward hooked function (cubic with three parameters).
    """
    x = jnp.linspace(-1.0, 1.0, 20)
    y = a * x**3 + b * x**2 + c * x
    
    # Return the reversed array since RARs are indexed from right to left (decreasing).
    return y[::-1]

def monotonic_fit(use_err:bool=True):
    a = sample("a", dist.Normal(2.0, 0.5))
    a_prior = sample("a (prior)", dist.Normal(2.0, 0.5))

    b = sample("b", dist.Normal(-3.0, 0.5))
    b_prior = sample("b (prior)", dist.Normal(-3.0, 0.5))

    c = sample("c", dist.Normal(-1.0, 0.5))
    c_prior = sample("c (prior)", dist.Normal(-1.0, 0.5))

    y = deterministic("y", hook_function_1d(a, b, c))
    y_prior = deterministic("y (prior)", hook_function_1d(a_prior, b_prior, c_prior))

    err = jnp.ones_like(y) * 0.5

    if use_err:
        ll = deterministic("log_likelihood", global_monotonic_ll(y, err))
        deterministic("ll (prior)", global_monotonic_ll(y_prior, err))
    else:
        ll = deterministic("log_likelihood", discrete_monotonic_ll(y))
        deterministic("ll (prior)", discrete_monotonic_ll(y_prior))

    factor("ll", ll)

def set_args():
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()
    parser = argparse.ArgumentParser(description="Mock hook")
    parser.add_argument("--progress-bar", default=True, type=bool, help="Shows MCMC progress bar.")
    parser.add_argument("--use-err", default=True, type=bool, help="Use errors in function and global ll (similar to gobs).")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = set_args()
    use_err = args.use_err

    nuts_kernel = NUTS(monotonic_fit, init_strategy=init_to_median(num_samples=1000))
    mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=20000, progress_bar=args.progress_bar)
    mcmc.run(random.PRNGKey(0), use_err=use_err)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    corner_samples = {
        "a": samples["a"],
        "b": samples["b"],
        "c": samples["c"],
        "-LL": jnp.clip(-samples["log_likelihood"], min=1e-10)
    }
    corner_priors = {
        "a (prior)": samples["a (prior)"],
        "b (prior)": samples["b (prior)"],
        "c (prior)": samples["c (prior)"],
        "-LL (prior)": jnp.clip(-samples["ll (prior)"], min=1e-10)
    }

    # min_ll = jnp.percentile(samples["ll (prior)"], 40.0)
    # range = [ 1.0, 1.0, 1.0, (min_ll, jnp.max(samples["log_likelihood"])) ]
    # print(range)
    
    sample_array = np.column_stack([np.array(corner_samples[key]) for key in corner_samples])
    figure = corner.corner(sample_array, labels=list(corner_samples.keys()), 
                           show_titles=True, color="tab:red", bins=40, axes_scale=['linear', 'linear', 'linear', 'log'])
    corner.corner(np.column_stack([np.array(corner_priors[key]) for key in corner_priors]), 
                  color="tab:blue", bins=40, fig=figure, axes_scale=['linear', 'linear', 'linear', 'log'])

    if use_err: figure.savefig("/mnt/users/koe/SPARC_RAR/plots/rm_hooks/mock_hook/mock_corner.png", dpi=300)
    else: figure.savefig("/mnt/users/koe/SPARC_RAR/plots/rm_hooks/mock_hook/mock_corner_discrete.png", dpi=300)
    plt.close(figure)
    print("Corner plot saved.")

    max_ll_idx = jnp.argmax(samples["log_likelihood"])
    x = jnp.linspace(-1.0, 1.0, 20)
    y = hook_function_1d()[::-1]
    y_best = hook_function_1d( samples['a'][max_ll_idx], samples['b'][max_ll_idx], samples['c'][max_ll_idx] )[::-1]

    if use_err:
        err = np.ones_like(y) * 0.5
        plt.errorbar(x, y, err, c='tab:blue', marker='.', capsize=2.0, label="Prior")
        plt.errorbar(x, y_best, err, c="tab:red", marker='.', capsize=2.0, label="Best fit")
        plt.savefig("/mnt/users/koe/SPARC_RAR/plots/rm_hooks/mock_hook/mock_hook.png", dpi=300)
    else:
        plt.plot(x, y,  c='tab:blue', marker='.', label="Prior")
        plt.plot(x, y_best, c="tab:red", marker='.', label="Best fit")
        plt.savefig("/mnt/users/koe/SPARC_RAR/plots/rm_hooks/mock_hook/mock_hook_discrete.png", dpi=300)
    plt.close()
