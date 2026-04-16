import numpy as np
from scipy.stats import gaussian_kde



def nd_posterior(samples, bins, weights=None):
    """
    samples : (Nsamples, N)
    bins    : list of N arrays of bin edges
    weights : optional weights (e.g. likelihoods)
    """
    hist, edges = np.histogramdd(
        samples,
        bins=bins,
        weights=weights,
        density=True
    )

    hist /= hist.sum()  # normalize to 1
    return hist, edges

def bin_centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])

def marginal_kd(samples, dims, bins, weights=None, density=True):
    """
    samples : (Nsamples, N)
    dims    : tuple of dimensions to keep (e.g. (0,2))
    bins    : list of bin edges for all dims
    """

    coords = [samples[:, d] for d in dims]
    use_bins = [bins[d] for d in dims]

    H, edges = np.histogramdd(
        np.vstack(coords).T,
        bins=use_bins,
        weights=weights,
        density=density
    )

    if density:
        H /= H.sum()

    return H, edges

def weighted_quantile(values, quantiles, weights):
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)







def hpd_interval(samples, cred=0.68):
    """
    Highest Posterior Density interval for 1D samples.
    """
    samples = np.sort(samples)
    n = len(samples)
    k = int(np.floor(cred * n))

    widths = samples[k:] - samples[:n - k]
    i = np.argmin(widths)

    return samples[i], samples[i + k]

def build_bins(samples, nbins=50, ranges=None):
    """
    samples : (Nsamples, N)
    ranges  : optional list of (min, max)
    """
    N = samples.shape[1]
    bins = []
    for i in range(N):
        lo, hi = min(samples[:, i]), max(samples[:, i])
        bins.append(np.linspace(lo, hi, nbins + 1))
    return bins

def bin_centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])

def emcee_weights(sampler, burnin=0, thin=1):
    logL = sampler.get_log_prob(
        discard=burnin,
        thin=thin,
        flat=True
    )
    return np.exp(logL - np.max(logL))

def plot_2d(ax, P, edges, **kwargs):
    x = 0.5 * (edges[0][1:] + edges[0][:-1])
    y = 0.5 * (edges[1][1:] + edges[1][:-1])
    ax.contourf(x, y, P.T, **kwargs)