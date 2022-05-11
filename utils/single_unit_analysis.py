import numpy as np
from task import generate_batch

from scipy.special import softmax
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

def get_FR(pos, firing_rates, bin_size, smooth=True):
    '''
    Compute the average and SEM firing rate by position for a given set
    of position and firing rate observations.

    Params
    ------
    pos : ndarray
        circular position at each time point; shape (batch_size * num_steps, 1)
    firing_rates : ndarray
        firing rate at each time point for each unit; shape (n_units, batch_size * num_steps)
    bin_size : float
        position bin size
    smooth : bool
        optional, if True tuning curve will be gaussian smoothed (sigma=2); default is True

    Returns
    -------
    firing_rate_avg : ndarray
        average firing rate in each position bin; shape (n_units, n_pos_bins)
    firing_rate_sem : ndarray
        SEM for the firing rate in each position bin; shape (n_units, n_pos_bins)
    '''
    n_units = firing_rates.shape[0]

    # define position bins
    edges = np.arange(np.min(pos), np.max(pos) + bin_size, bin_size)
    b_idx = np.digitize(pos, edges)
    unique_bdx = np.unique(b_idx)

    # find FR in each bin
    firing_rate_avg = np.zeros((n_units, unique_bdx.shape[0]))
    firing_rate_sem = np.zeros((n_units, unique_bdx.shape[0]))
    for  i, b in enumerate(unique_bdx):
        firing_rate_sem[:, i] = stats.sem(firing_rates[:, b_idx == b], axis=1)
        firing_rate_avg[:, i] = np.mean(firing_rates[:, b_idx == b], axis=1)
    if smooth:
        firing_rate_avg = gaussian_filter1d(firing_rate_avg, 2, axis=1, mode='wrap')
        fr_sem = gaussian_filter1d(firing_rate_sem, 2, axis=1, mode='wrap')
        
    return firing_rate_avg, firing_rate_sem