#!/usr/bin/env python
# Attempt to remove hooks from the SPARC RAR.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import jax
from jax import numpy as jnp
import jax.random as random
from jax.scipy.stats import norm
from jax.scipy.special import log_ndtr

import numpyro
from numpyro import distributions as dist
from numpyro import deterministic, sample, factor
import corner

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_analysis.params import pdisk, pbul
from utils_analysis.get_SPARC import get_SPARC_data
from numpyro.infer import MCMC, NUTS, init_to_median

def vel2acc(vel_sq, rad):
    """
    Calculate the gravitational acceleration (g_obs or g_bar).
    In the data, vel is in km/s, and rad is in kpc.
    Returns g_obs in m/s².
    """
    return (vel_sq * 1e6) / (rad * 3.086e19)    # Convert to m/s²

def errV2errA(vel, vel_err, rad):
    """
    Convert Gaussian errors in velocity to errors in acceleration.
    vel: velocity in km/s
    vel_err: error in velocity in km/s
    rad: radius in kpc
    Returns error in acceleration in m/s².
    """
    # g = v^2 / r (with unit conversions)
    # Error propagation: dg = 2*v*dv / r (with unit conversions)
    return (2 * vel * vel_err * 1e6) / (rad * 3.086e19)

def monotonic_ll(mu, sigma):
    """
    mu: array of means [mu_1, mu_2, ..., mu_n]
    sigma: array of std deviations [sigma_1, ..., sigma_n]
    
    Returns log-likelihood that the sequence is (pairwise) monotonically increasing.
    """
    diffs = mu[1:] - mu[:-1]
    denom = jnp.sqrt(sigma[1:]**2 + sigma[:-1]**2)
    z = diffs / denom
    probs = norm.cdf(z)     # Use standard normal CDF (creds to Richard for derivation)
    
    logL = jnp.sum(jnp.log(probs + 1e-12))  # Avoid log(0) with small epsilon
    return logL

def global_monotonic_ll(mu, sigma):
    """
    mu: array of means [mu_1, mu_2, ..., mu_n]
    sigma: array of std deviations [sigma_1, ..., sigma_n]

    Returns: scalar log-likelihood preferring x_m > x_n for all m > n (global monotonicity).
    """
    # Pairwise differences and denominators via broadcasting
    dmu = mu[:, None] - mu[None, :]                     # shape (n, n)
    denom = jnp.sqrt(sigma[:, None]**2 + sigma[None, :]**2) # (n, n)

    # z-scores; mask diagonal / lower triangle
    z = dmu / (denom + 1e-30)                           # avoid division by zero
    mask = jnp.tril(jnp.ones_like(z, dtype=bool), k=-1) # m > n -> lower triangle

    # Sum log Phi only over m > n
    return jnp.sum(jnp.where(mask, log_ndtr(z), 0.0))

def g_obs_fit(table, i_table, data, bulged, pdisk:float=pdisk, pbul:float=pbul):
    # Sample mass-to-light ratios.
    smp_pgas  = sample("Gas M/L", dist.TruncatedNormal(1.0, 0.09, low=0.0))
    smp_pdisk = sample("Disk M/L", dist.TruncatedNormal(pdisk, 0.125, low=0.0))
    if bulged: smp_pbul = sample("Bulge M/L", dist.TruncatedNormal(pbul, 0.175, low=0.0))
    else: smp_pbul = deterministic("Bulge M/L", jnp.array(0.0))

    # Sample luminosity.
    L = sample("L", dist.TruncatedNormal(table["L"][i_table], table["e_L"][i_table], low=0.0))
    smp_pdisk *= L / table["L"][i_table]
    smp_pbul *= L / table["L"][i_table]

    # Sample inclination (convert from degrees to radians!) and scale Vobs accordingly
    inc_min, inc_max = 15 * jnp.pi / 180, 150 * jnp.pi / 180
    inc = sample("inc",dist.TruncatedNormal(table["Inc"][i_table]*jnp.pi/180, table["e_Inc"][i_table]*jnp.pi/180, low=inc_min, high=inc_max))
    inc_scaling = jnp.sin(table["Inc"][i_table]*jnp.pi/180) / jnp.sin(inc)

    Vobs = deterministic("Vobs", jnp.array(data["Vobs"]) * inc_scaling)
    e_Vobs = deterministic("e_Vobs", jnp.array(data["errV"]) * inc_scaling)

    # Sample distance to the galaxy.
    d = sample("Distance", dist.TruncatedNormal(table["D"][i_table], table["e_D"][i_table], low=0.0))
    d_scaling = d / table["D"][i_table]

    if bulged:
        Vbar_squared = (jnp.array(data["Vgas"]**2) * smp_pgas + 
                        jnp.array(data["Vdisk"]**2) * smp_pdisk + 
                        jnp.array(data["Vbul"]**2) * smp_pbul)
    else:
        Vbar_squared = (jnp.array(data["Vgas"]**2) * smp_pgas + 
                        jnp.array(data["Vdisk"]**2) * smp_pdisk)
    Vbar_squared *= d_scaling
    Vbar = deterministic("Vbar", jnp.sqrt(Vbar_squared))

    # Sample parameters and calculate the predicted g_obs.
    r = deterministic("r", jnp.array(data["Rad"]) * d_scaling)
    errA = deterministic("errA", errV2errA(jnp.sqrt(Vobs), e_Vobs, r))
    g_bar = deterministic("g_bar", vel2acc(Vbar_squared, r))
    g_obs = deterministic("g_obs", vel2acc(Vobs**2, r))
    
    # Likelihood: monotonicity constraint on the RAR.
    ll = monotonic_ll(g_obs, errA)

    deterministic("log_likelihood", ll)
    factor("ll", ll)

def set_args():
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()
    parser = argparse.ArgumentParser(description="MCMC fit for unhooking RARs")
    parser.add_argument("--progress-bar", default=False, type=bool, help="Show MCMC progress bar.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = set_args()

    SPARC_data, _, _ = get_SPARC_data()
    galaxies = [
        "D564-8", "UGC00731", "D631-7", "UGC04278", "DDO154",
        "UGC05414", "DDO168", "UGC05764", "ESO116-G012", "UGC05986",
        "F574-1", "UGC06667", "IC2574", "UGC06917", "KK98-251",
        "UGC07089", "NGC0055", "UGC07151", "NGC0100", "UGC07399",
        "NGC2403", "UGC07603", "NGC3109", "UGC08837", "NGC4010", "UGCA442"
        ]   # 26 'hooked' galaxies from https://arxiv.org/pdf/2307.09507

    # Get more info from SPARC table.
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"
    SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    table = pd.read_fwf(file, skiprows=98, names=SPARC_c)

    unhooked_RAR = {}
    for i, gal in enumerate(galaxies):
        print(f"Processing galaxy {gal} ({i+1}/26)...")
        i_table = np.where(table["Galaxy"] == gal)[0][0]
        
        nuts_kernel = NUTS(g_obs_fit, init_strategy=init_to_median(num_samples=1000))
        mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=20000, progress_bar=args.progress_bar)
        mcmc.run(random.PRNGKey(0), table, i_table, SPARC_data[gal]["data"], SPARC_data[gal]["bulged"])
        mcmc.print_summary()
        samples = mcmc.get_samples()

        # Plot corner plot for MCMC samples
        corner_samples = {
            "Gas M/L": samples["Gas M/L"],
            "Disk M/L": samples["Disk M/L"],
            "Bulge M/L": samples["Bulge M/L"],
            "inc": samples["inc"],
            "Distance": samples["Distance"],
            "L": samples["L"],
        }
        # Only include Bulge M/L if bulged
        if not SPARC_data[gal]["bulged"]:
            del corner_samples["Bulge M/L"]

        sample_array = np.column_stack([np.array(corner_samples[key]) for key in corner_samples])
        figure = corner.corner(sample_array, labels=list(corner_samples.keys()), show_titles=True)
        figure.savefig(f"/mnt/users/koe/SPARC_RAR/plots/rm_hooks/corner_plots/{gal}.png", dpi=300)
        plt.close(figure)

        log_likelihood = samples["log_likelihood"]
        max_ll_idx = jnp.argmax(log_likelihood)
        g_bar = samples["g_bar"][max_ll_idx]
        g_obs = samples["g_obs"][max_ll_idx]
        Vbar = samples["Vbar"][max_ll_idx]
        Vobs = samples["Vobs"][max_ll_idx]

        unhooked_RAR[gal] = {
            'g_bar': g_bar,
            'g_obs': g_obs,
            'Vbar': Vbar,
            'Vobs': Vobs,
        }

        # Plot RAR with best-fit parameters.
        plt.scatter(SPARC_data[gal]["g_bar"], SPARC_data[gal]["g_obs"], color='k', alpha=0.5, label='data')
        plt.scatter(g_bar, g_obs, color='tab:red', label='max likelihood')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('g_bar [m/s²]')
        plt.ylabel('g_obs [m/s²]')
        plt.title(f"{gal} RAR (log-log)")
        plt.legend()
        plt.savefig(f"/mnt/users/koe/SPARC_RAR/plots/rm_hooks/{gal}.png", dpi=300)
        plt.close()

        jax.clear_caches()

    np.save(f"/mnt/users/koe/SPARC_RAR/unhooked_RAR.npy", unhooked_RAR)  # Save the unhooked RARs.
