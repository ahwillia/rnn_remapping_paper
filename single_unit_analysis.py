import sys
sys.path.append("../utils/")
import numpy as np
from task import generate_batch

from scipy.special import softmax
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance

from sklearn.decomposition import PCA
from tqdm import trange

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

def get_chunked_FR(pos_targets, firing_rates, bin_size, n_splits=100):
    '''
    Compute the average firing rate by position for each batch separately.
    
    Params
    ------
    pos_targets : ndarray
        circular position; shape (num_steps * batch_size)
    firing_rates : ndarray
        firing rates for each unit; shape (n_units, num_steps * batch_size)
    n_splits : int
        number of chunks to split the data into
        (should be small enough to get full pos coverage in each chunk)

    Returns
    -------
    split_FR_avg : ndarray
        average firing rate in each position bin for each batch; shape (n_units, n_pos_bins, n_splits)
    '''
    T = pos_targets.shape[0]
    n_units = firing_rates.shape[0]
    edges = np.arange(np.min(pos_targets), np.max(pos_targets) + bin_size, bin_size)
    n_pos_bins = edges.shape[0] - 1

    # get index to split the data into chunks
    full_idx = np.arange(T)
    split_idx = np.array_split(full_idx, n_splits)
    
    # calculate the average firing rate for each chunk
    split_FR_avg = np.zeros([n_units, n_pos_bins, n_splits])
    for i in trange(n_splits):
        idx = split_idx[i]
        pos = pos_targets[idx]
        fr = firing_rates[:, idx]
        split_FR_avg[:, :, i], _ = get_FR(pos=pos, firing_rates=fr, bin_size=bin_size)
    
    return split_FR_avg

def spatial_info(pos_targets, firing_rates, bin_size):
    '''
    Compute spatial information (bits per second) averaged across all trials
    [seems low? possibly off by a factor of 10 or something]
    
    Params:
    -------
    pos_targets : ndarray
        circular position at each time point; shape (batch_size * num_steps, 1)
    firing_rates : ndarray
        firing rates for each unit; shape (n_units, batch_size * num_steps)
    bin_size : float
        position bin size (rad)
        
    Returns:
    -------
    SI : ndarray
        spatial information for each unit; shape (n_units, 1)
    '''
    n_units = firing_rates.shape[0]

    # get position bins
    edges = np.arange(np.min(pos_targets), np.max(pos_targets) + bin_size, bin_size)
    b_idx = np.digitize(pos_targets, edges)
    unique_bdx = np.unique(b_idx)
    
    # get params
    T = pos_targets.shape[0] # total time
    L = np.sum(firing_rates, axis=1) / T # overall mean FR
    
    SI = np.zeros(n_units) # spatial info
    for i, b in enumerate(unique_bdx):
        occupancy = np.sum(b_idx == b)
        t = np.sum(b_idx == b) / T # occupancy ratio (time in bin/total time)
        l = np.mean(firing_rates[:, b_idx == b], axis=1) # mean FR for that bin
        SI += t * l * np.log2((l / L) + 0.0001)
    SI[L == 0] = 0
        
    return SI

def coding_stability(pos_targets, firing_rates, **kwargs):
    '''
    Compute the spatial correlation across chunks of batches for each unit.
    
    Params
    ------
    pos_targets : ndarray
        circular position at each timepoint; shape (num_steps * batch_size)
    firing_rates : ndarray
        firing rates for each unit; shape (n_units, num_steps * batch_size)
    **kwargs : passed to get_chunked_FR

    Returns
    -------
    split_FR_avg : ndarray
        average firing rate in each position bin for each batch; shape (n_units, n_pos_bins, n_splits)
    '''
    n_units = firing_rates.shape[0]

    # get the avg firing rate for different chunks of the data
    split_FR_avg = get_chunked_FR(pos_targets, firing_rates, **kwargs)
    
    # get the cross-correlation for each unit
    avg_correlation = np.zeros(n_units)
    for u in range(n_units):
        fr = split_FR_avg[u, :, :].copy()
        sim_vec = np.abs(distance.pdist(fr.T, 'correlation')-1)
        avg_correlation[u] = np.mean(sim_vec)
        
    return avg_correlation

''' characterize single unit remapping properties '''
def pct_change_FR(avg_FR_1, avg_FR_2):
    ''' find the absolute percent difference in firing rates '''
    peak_FR_1 = np.max(avg_FR_1, axis=1)
    peak_FR_2 = np.max(avg_FR_2, axis=1)
    return (np.abs(peak_FR_2 - peak_FR_1) / ((peak_FR_2 + peak_FR_1)/2))*100

def spatial_dissimilarity(avg_FR_1, avg_FR_2):
    '''
    find how dissimilar 2 sets of firing rates are
    dissimilarity = 1 - cosine similarity
    '''
    n_units = avg_FR_1.shape[0]
    
    # calculate cosine similarity
    norms_1 = np.linalg.norm(avg_FR_1, axis=1)
    norms_2 = np.linalg.norm(avg_FR_2, axis=1)
    angle = np.zeros(n_units)
    for i, n1 in enumerate(norms_1):
        n2 = norms_2[i]
        normalized_FR_1 = avg_FR_1[i, :]/n1
        normalized_FR_2 = avg_FR_2[i, :]/n2   
        angle[i] = normalized_FR_1 @ normalized_FR_2
        
    # return dissimilarity
    return 1 - angle

def pct_on_off(avg_FR_1, avg_FR_2, thresh=0.001):
    n_units = avg_FR_1.shape[0]
    
    # get the max firing rate in each map for each cell
    peak_FR_1 = np.max(avg_FR_1, axis=1)
    peak_FR_2 = np.max(avg_FR_2, axis=1)

    # get number of cells that are off in one map
    off_1 = peak_FR_1 < thresh
    off_2 = peak_FR_2 < thresh
    on_off = (off_1 & ~off_2) | (~off_1 & off_2)
    
    return (np.sum(on_off) / n_units) * 100