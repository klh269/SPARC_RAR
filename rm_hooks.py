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
from numpyro.distributions.transforms import AffineTransform
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
    return (vel_sq * 1e6) / (rad * 3.086e19)

def errV2errA(vel, vel_err, rad):
    """
    Convert Gaussian errors in velocity to (approximately Gaussian) errors in acceleration.
    vel: velocity in km/s
    vel_err: error in velocity in km/s
    rad: radius in kpc

    Returns error in acceleration in m/s².
    """
    # sample_shape = (num_samples,) + vel.shape
    # vel_samples = random.normal(key, sample_shape) * vel_err + vel
    # acc_samples = (vel_samples * 1e3)**2 / (rad * 3.086e19)
    # return jnp.std(acc_samples, axis=0)

    # MC sampling is way too inefficient, let's just assume vel_err << vel...
    return ( 2 * vel * vel_err * 1e6 ) / ( rad * 3.086e19 )


def transformed_normal(loc, scale):
    """
    Log-normal prior for SPARC mass-to-light ratios.
    """
    return dist.TransformedDistribution( dist.Normal(0, 1), AffineTransform(loc=loc, scale=scale) )

def global_monotonic_ll(mu, sigma):
    """
    mu: array of means [mu_1, mu_2, ..., mu_n]
    sigma: array of std deviations [sigma_1, ..., sigma_n]

    Returns: scalar log-likelihood preferring x_m < x_n for all m > n (global monotonicity).
    NOTE: points are arranged by radii, so we want monotonically decreasing functions!
    """
    # Pairwise differences and denominators via broadcasting
    dmu = mu[:, None] - mu[None, :]                         # shape (n, n)
    denom = jnp.sqrt(sigma[:, None]**2 + sigma[None, :]**2) # (n, n)

    # z-scores; mask diagonal / upper triangle
    z = dmu / (denom + 1e-30)                           # avoid division by zero
    mask = jnp.triu(jnp.ones_like(z, dtype=bool), k=1)  # m < n -> upper triangle

    # Sum log Phi only over m < n
    return jnp.sum(jnp.where(mask, log_ndtr(z), 0.0))

def discrete_monotonic_ll(arr):
    """
    arr: an array of deterministic values [a_1, a_2, ..., a_n]

    Returns: log-likelihood that penalizes pairwise non-monotonicity for all valid pairs.
    """
    n = len(arr)

    # Create difference matrix: diff[i,j] = arr[i] - arr[j]
    diff = arr[:, jnp.newaxis] - arr[jnp.newaxis, :]
    
    # Get upper triangle (i < j)
    i_idx, j_idx = jnp.triu_indices(n, k=1)
    differences = diff[i_idx, j_idx]
    
    score = ( jnp.sum(differences >= 0) + 1 ) / ( len(differences) + 1 )    # +1s inserted to avoid log(0).
    return jnp.log(score)


def g_obs_fit(table, i_table, data, bulged, pdisk:float=pdisk, pbul:float=pbul):
    # Sample mass-to-light ratios.
    log_pgas = sample("Gas M/L (log)", transformed_normal(jnp.log10(1.), 0.04))
    prior_log_pgas = sample("Gas M/L (prior)", transformed_normal(jnp.log10(1.), 0.04))     # Track prior for later plots.

    log_pdisk = sample("Disk M/L (log)", transformed_normal(jnp.log10(pdisk), 0.1))
    prior_log_pdisk = sample("Disk M/L (prior)", transformed_normal(jnp.log10(pdisk), 0.1))

    if bulged:
        log_pbul = sample("Bulge M/L (log)", transformed_normal(jnp.log10(pbul), 0.1))
        prior_log_pbul = sample("Bulge M/L (prior)", transformed_normal(jnp.log10(pbul), 0.1))
    else:
        log_pbul = deterministic("Bulge M/L (log)", jnp.array(0.0))
        prior_log_pbul = deterministic("Bulge M/L (prior)", jnp.array(0.0))

    smp_pgas, smp_pdisk, smp_pbul = 10**log_pgas, 10**log_pdisk, 10**log_pbul
    prior_smp_pgas, prior_smp_pdisk, prior_smp_pbul = 10**prior_log_pgas, 10**prior_log_pdisk, 10**prior_log_pbul

    # Sample luminosity.
    L = sample("L", dist.TruncatedNormal(table["L"][i_table], table["e_L"][i_table], low=0.0))
    prior_L = sample("L (prior)", dist.TruncatedNormal(table["L"][i_table], table["e_L"][i_table], low=0.0))
    smp_pdisk *= L / table["L"][i_table]
    prior_smp_pdisk *= prior_L / table["L"][i_table]
    smp_pbul *= L / table["L"][i_table]
    prior_smp_pbul *= prior_L / table["L"][i_table]

    # Sample inclination (convert from degrees to radians!) and scale Vobs accordingly
    inc_min, inc_max = 15 * jnp.pi / 180, 150 * jnp.pi / 180
    inc = sample("inc", dist.TruncatedNormal(table["Inc"][i_table]*jnp.pi/180, table["e_Inc"][i_table]*jnp.pi/180, low=inc_min, high=inc_max))
    prior_inc = sample("inc (prior)", dist.TruncatedNormal(table["Inc"][i_table]*jnp.pi/180, table["e_Inc"][i_table]*jnp.pi/180, low=inc_min, high=inc_max))
    inc_scaling = jnp.sin(table["Inc"][i_table]*jnp.pi/180) / jnp.sin(inc)
    prior_inc_scaling = jnp.sin(table["Inc"][i_table]*jnp.pi/180) / jnp.sin(prior_inc)

    Vobs = deterministic("Vobs", jnp.array(data["Vobs"]) * inc_scaling)
    prior_Vobs = deterministic("Vobs (prior)", jnp.array(data["Vobs"]) * prior_inc_scaling)
    e_Vobs = deterministic("e_Vobs", jnp.array(data["errV"]) * inc_scaling)
    prior_e_Vobs = deterministic("e_Vobs (prior)", jnp.array(data["errV"]) * prior_inc_scaling)

    # Sample distance to the galaxy.
    d = sample("Distance", dist.TruncatedNormal(table["D"][i_table], table["e_D"][i_table], low=1e-10))
    prior_d = sample("Distance (prior)", dist.TruncatedNormal(table["D"][i_table], table["e_D"][i_table], low=1e-10))
    d_scaling = d / table["D"][i_table]
    prior_d_scaling = prior_d / table["D"][i_table]

    if bulged:
        Vbar_squared = (jnp.array(data["Vgas"]**2) * smp_pgas + 
                        jnp.array(data["Vdisk"]**2) * smp_pdisk + 
                        jnp.array(data["Vbul"]**2) * smp_pbul)
        prior_Vbar_squared = (jnp.array(data["Vgas"]**2) * prior_smp_pgas +
                              jnp.array(data["Vdisk"]**2) * prior_smp_pdisk +
                              jnp.array(data["Vbul"]**2) * prior_smp_pbul)
    else:
        Vbar_squared = (jnp.array(data["Vgas"]**2) * smp_pgas + 
                        jnp.array(data["Vdisk"]**2) * smp_pdisk)
        prior_Vbar_squared = (jnp.array(data["Vgas"]**2) * prior_smp_pgas +
                              jnp.array(data["Vdisk"]**2) * prior_smp_pdisk)
    Vbar_squared *= d_scaling
    prior_Vbar_squared *= prior_d_scaling
    deterministic("Vbar", jnp.sqrt(Vbar_squared))

    # Sample parameters and calculate the predicted g_obs.
    r = deterministic("r", jnp.array(data["Rad"]) * d_scaling)
    g_bar = deterministic("g_bar", vel2acc(Vbar_squared, r))
    g_obs = deterministic("g_obs", vel2acc(Vobs**2, r))

    prior_r = deterministic("r (prior)", jnp.array(data["Rad"]) * prior_d_scaling)
    prior_g_bar = deterministic("g_bar (prior)", vel2acc(prior_Vbar_squared, prior_r))
    prior_g_obs = deterministic("g_obs (prior)", vel2acc(prior_Vobs**2, prior_r))

    e_gobs = errV2errA(Vobs, e_Vobs, r)
    prior_e_gobs = errV2errA(prior_Vobs, prior_e_Vobs, prior_r)

    # Track original log-likelihood for reference.
    prior_ll_gbar = discrete_monotonic_ll(prior_g_bar)
    prior_ll_gobs = global_monotonic_ll(prior_g_obs, prior_e_gobs)

    # Small variation to artificially create dynamic range for corner plots whenever necessary.
    deterministic("prior_ll_gobs", prior_ll_gobs)
    deterministic("prior_ll_gbar", prior_ll_gbar)
    # prior_ll_gbar = sample("prior_ll_gbar", dist.Normal(prior_ll_gbar, jnp.abs(prior_ll_gbar/1e3)))
    # prior_ll_gobs = sample("prior_ll_gobs", dist.Normal(prior_ll_gobs, jnp.abs(prior_ll_gobs/1e3)))
    deterministic("prior_ll", prior_ll_gbar + prior_ll_gobs)

    # Likelihood: monotonicity constraint on the RAR (probabilistic for g_obs and deterministic for g_bar).
    ll_gbar = discrete_monotonic_ll(g_bar)
    ll_gobs = global_monotonic_ll(g_obs, e_gobs)
    # ll = deterministic("log_likelihood", ll_gobs)   # Testing without considering monotonicity in g_bar.

    deterministic("log_likelihood_gobs", ll_gobs)
    deterministic("log_likelihood_gbar", ll_gbar)
    # ll_gbar = sample("log_likelihood_gbar", dist.Normal(ll_gbar, jnp.abs(ll_gbar/1e3)))
    # ll_gobs = sample("log_likelihood_gobs", dist.Normal(ll_gobs, jnp.abs(ll_gobs/1e3)))

    ll = deterministic("log_likelihood", ll_gbar + ll_gobs)

    factor("ll", ll)


def set_args():
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()
    parser = argparse.ArgumentParser(description="MCMC for unhooking RARs")
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
        print(f"\nProcessing galaxy {gal} ({i+1}/26)...")
        i_table = np.where(table["Galaxy"] == gal)[0][0]
        r = jnp.array(SPARC_data[gal]["r"])
        data = SPARC_data[gal]["data"]
        bulged = SPARC_data[gal]["bulged"]

        nuts_kernel = NUTS(g_obs_fit, init_strategy=init_to_median(num_samples=1000))
        mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=20000, progress_bar=args.progress_bar)
        mcmc.run(random.PRNGKey(0), table, i_table, data, bulged)
        mcmc.print_summary()
        samples = mcmc.get_samples()

        # Plot corner plot for MCMC samples
        # Specify ranges for log-likelihoods (mostly fixed, i.e., RARs can't be unhooked...)
        min_smp_ll_bar, max_smp_ll_bar = jnp.min( samples["log_likelihood_gbar"] ), jnp.max( samples["log_likelihood_gbar"] )
        min_ll_bar = min_smp_ll_bar - max( ( max_smp_ll_bar - min_smp_ll_bar ) / 2, abs(min_smp_ll_bar) / 10 )
        max_ll_bar = max_smp_ll_bar + max( ( max_smp_ll_bar - min_smp_ll_bar ) / 2, abs(min_smp_ll_bar) / 10 )

        min_smp_ll_obs, max_smp_ll_obs = jnp.min( samples["log_likelihood_gobs"] ), jnp.max( samples["log_likelihood_gobs"] )
        min_ll_obs = min_smp_ll_obs - max( ( max_smp_ll_obs - min_smp_ll_obs ) / 2, abs(min_smp_ll_obs) / 10 )
        max_ll_obs = max_smp_ll_obs + max( ( max_smp_ll_obs - min_smp_ll_obs ) / 2, abs(min_smp_ll_obs) / 10 )

        min_smp_ll, max_smp_ll = jnp.min( samples["log_likelihood"] ), jnp.max( samples["log_likelihood"] )
        min_ll = min_smp_ll - max( ( max_smp_ll - min_smp_ll ) / 2, abs(min_smp_ll) / 10 )
        max_ll = max_smp_ll + max( ( max_smp_ll - min_smp_ll ) / 2, abs(min_smp_ll) / 10 )

        corner_samples = {
            "Gas M/L (log)": samples["Gas M/L (log)"],
            "Disk M/L (log)": samples["Disk M/L (log)"],
            "Bulge M/L (log)": samples["Bulge M/L (log)"],
            "inc": samples["inc"],
            "Distance": samples["Distance"],
            "L": samples["L"],
            "LL (g_bar)": samples["log_likelihood_gbar"],
            "LL (g_obs)": samples["log_likelihood_gobs"],
            "LL (all)": samples["log_likelihood"]
        }
        corner_priors = {
            "Gas M/L (log)": samples["Gas M/L (prior)"],
            "Disk M/L (log)": samples["Disk M/L (prior)"],
            "Bulge M/L (log)": samples["Bulge M/L (prior)"],
            "inc": samples["inc (prior)"],
            "Distance": samples["Distance (prior)"],
            "L": samples["L (prior)"],
            "Prior LL (g_bar)": samples["prior_ll_gbar"],
            "Prior LL (g_obs)": samples["prior_ll_gobs"],
            "Prior LL (all)": samples["prior_ll"]
        }
        range = [ 1., 1., 1., 1., 1., 1., (min_ll_bar, max_ll_bar), (min_ll_obs, max_ll_obs), (min_ll, max_ll) ]
        # Only include Bulge M/L if bulged
        if not SPARC_data[gal]["bulged"]:
            corner_samples.pop("Bulge M/L (log)")
            corner_priors.pop("Bulge M/L (log)")
            range = range[1:]

        sample_array = np.column_stack([np.array(corner_samples[key]) for key in corner_samples])
        figure = corner.corner(np.column_stack([np.array(corner_priors[key]) for key in corner_priors]), show_titles=True, color="tab:blue", range=range, bins=40)
        corner.corner(sample_array, labels=list(corner_samples.keys()), show_titles=True, color="tab:red", fig=figure, range=range, bins=40)

        figure.savefig(f"/mnt/users/koe/SPARC_RAR/plots/rm_hooks/corner_plots/{gal}.png", dpi=300)
        plt.close(figure)
        print("Corner plots saved.")

        # Create individual plots to compare RARs before and after fits.
        log_likelihood = samples["log_likelihood"]
        max_ll_idx = jnp.argmax(log_likelihood)
        g_bar = samples["g_bar"][max_ll_idx]
        g_obs = samples["g_obs"][max_ll_idx]
        Vbar = samples["Vbar"][max_ll_idx]
        Vobs = samples["Vobs"][max_ll_idx]
        errV = samples["e_Vobs"][max_ll_idx]

        unhooked_RAR[gal] = {
            'g_bar': g_bar,
            'g_obs': g_obs,
            'Vbar': Vbar,
            'Vobs': Vobs,
            'errV': errV
        }

        # Plot RAR with best-fit parameters.
        plt.plot(SPARC_data[gal]["g_bar"], SPARC_data[gal]["g_obs"], marker='.', color='k', alpha=0.5, label='data')
        plt.plot(g_bar, g_obs, marker='.', color='tab:red', label='max LL')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('g_bar [m/s²]')
        plt.ylabel('g_obs [m/s²]')
        plt.title(f"{gal} RAR (log-log)")
        plt.legend()
        plt.savefig(f"/mnt/users/koe/SPARC_RAR/plots/rm_hooks/{gal}.png", dpi=300)
        plt.close()

        print("Individual RAR plot saved.")
        jax.clear_caches()

    np.save(f"/mnt/users/koe/SPARC_RAR/unhooked_RAR.npy", unhooked_RAR)  # Save the unhooked RARs.

### Old functions --- no longer in use ###
# def estimate_covariance(table, i_table, data, bulged, num_samples=10000, pdisk:float=pdisk, pbul:float=pbul):
#     """
#     Estimate the covariance matrix of g_obs and g_bar by sampling the priors.

#     Returns: cov_matrix (2x2 numpy array), means (length-2 numpy array)
#     NOTE: Not needed... covariance is accounted for in the MCMC samples...
#     """
#     rng = np.random.default_rng()

#     # Sample priors
#     pgas_samples = rng.normal(1.0, 0.09, num_samples)
#     pgas_samples = np.clip(pgas_samples, 0.0, None)

#     pdisk_samples = rng.normal(pdisk, 0.125, num_samples)
#     pdisk_samples = np.clip(pdisk_samples, 0.0, None)

#     if bulged:
#         pbul_samples = rng.normal(pbul, 0.175, num_samples)
#         pbul_samples = np.clip(pbul_samples, 0.0, None)
#     else:
#         pbul_samples = np.zeros(num_samples)

#     L_samples = rng.normal(table["L"][i_table], table["e_L"][i_table], num_samples)
#     L_samples = np.clip(L_samples, 0.0, None)

#     pdisk_samples *= L_samples / table["L"][i_table]
#     pbul_samples *= L_samples / table["L"][i_table]

#     inc_samples = rng.normal(table["Inc"][i_table]*np.pi/180, table["e_Inc"][i_table]*np.pi/180, num_samples)
#     inc_samples = np.clip(inc_samples, 15*np.pi/180, 150*np.pi/180)
#     inc_scaling = np.sin(table["Inc"][i_table]*np.pi/180) / np.sin(inc_samples)

#     Vobs = np.array(data["Vobs"])
#     errV = np.array(data["errV"])
#     Vobs_samples = Vobs[None, :] * inc_scaling[:, None]     # Each row is a Vobs sample with one inclination scaling
#     errV_samples = errV[None, :] * inc_scaling[:, None]
#     Vobs_samples += rng.normal(0, 1, Vobs_samples.shape) * errV_samples

#     d_samples = rng.normal(table["D"][i_table], table["e_D"][i_table], num_samples)
#     d_samples = np.clip(d_samples, 1e-20, None)
#     d_scaling = d_samples / table["D"][i_table]

#     Vgas = np.array(data["Vgas"])
#     Vdisk = np.array(data["Vdisk"])
#     Vbul = np.array(data["Vbul"]) if bulged else np.zeros_like(Vdisk)
#     r = np.array(data["Rad"])

#     Vbar_squared_samples = (
#         Vgas[None, :]**2 * pgas_samples[:, None] +
#         Vdisk[None, :]**2 * pdisk_samples[:, None] +
#         Vbul[None, :]**2 * pbul_samples[:, None]
#     ) * d_scaling[:, None]

#     r_samples = r[None, :] * d_scaling[:, None]
#     g_bar_samples = vel2acc(Vbar_squared_samples, r_samples)
#     g_obs_samples = vel2acc(Vobs_samples**2, r_samples)

#     cov_g_bar = jnp.cov(g_bar_samples, rowvar=False)
#     cov_g_obs = jnp.cov(g_obs_samples, rowvar=False)

#     # print(f"Dimensions of covariance matrices: {cov_g_bar.shape}, {cov_g_obs.shape}")
#     # print(f"No. of data points: {len(data['Rad'])}")
#     # raise NotImplementedError("Testing covariance estimation, stopping here.")

#     return cov_g_bar, cov_g_obs

# def monotonic_ll(mu, sigma):
#     """
#     mu: array of means [mu_1, mu_2, ..., mu_n]
#     sigma: array of std deviations [sigma_1, ..., sigma_n]
    
#     Returns log-likelihood that the sequence is (pairwise) monotonically increasing.
#     """
#     diffs = mu[:-1] - mu[1:]
#     denom = jnp.sqrt(sigma[1:]**2 + sigma[:-1]**2)
#     z = diffs / denom
#     probs = norm.cdf(z)     # Use standard normal CDF (creds to Richard for derivation)
    
#     logL = jnp.sum(jnp.log(probs + 1e-12))  # Avoid log(0) with small epsilon
#     return logL

# def global_monotonic_ll_cov(mu, cov):
#     """
#     mu: array of means [mu_1, mu_2, ..., mu_n]
#     cov: covariance matrix (n x n) for the uncertainties in mu

#     Returns: scalar log-likelihood preferring x_m > x_n for all m > n (global monotonicity),
#     accounting for correlated uncertainties with the full covariance matrix.
#     NOTE: Again not needed since covariance is accounted for in the MCMC samples...
#     """
#     # Pairwise differences
#     dmu = mu[:, None] - mu[None, :]  # shape (n, n)

#     # Pairwise variances: Var(x_m - x_n) = cov[m, m] + cov[n, n] - 2 * cov[m, n]
#     var_diff = cov.diagonal()[:, None] + cov.diagonal()[None, :] - 2 * cov

#     # Avoid negative variances due to numerical issues
#     var_diff = jnp.clip(var_diff, a_min=1e-30, a_max=None)
#     std_diff = jnp.sqrt(var_diff)

#     # z-scores; mask diagonal / lower triangle
#     z = dmu / std_diff
#     mask = jnp.tril(jnp.ones_like(z, dtype=bool), k=-1)  # m > n -> lower triangle

#     # Sum log Phi only over m > n
#     return jnp.sum(jnp.where(mask, log_ndtr(z), 0.0))
