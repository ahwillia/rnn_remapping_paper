import numpy as np
from scipy import stats
from scipy.spatial import distance as dist
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

# Methods from Low et al, 2021 -- see STAR Methods for more details.

# single cell analysis
def tuning_curve(x, Y, dt, b, smooth=True, l=2, SEM=False):
    '''
    Params
    ------
    x : ndarray
        variable of interest by observation; shape (n_obs, )
    Y : ndarray
        spikes per observation; shape (n_obs, n_cells)
    dt : int
        time per observation in seconds
    b : int
        bin size
    smooth : bool
        apply gaussian filter to firing rate; optional, default is True
    l : int
        smoothness param for gaussian filter; optional, default is 2
    SEM : bool
        return SEM for FR; optional, default is False

    Returns
    -------
    firing_rate : ndarray
        trial-averaged, binned firing rate for each cell
        shape (n_bins, n_cells)
    centers : ndarray
        center of each bin
    '''
    edges = np.arange(0, np.max(x) + b, b)
    centers = (edges[:-1] + edges[1:])/2
    b_idx = np.digitize(x, edges)
    if np.max(x) == edges[-1]:
        b_idx[b_idx==np.max(b_idx)] = np.max(b_idx) - 1
    unique_bdx = np.unique(b_idx)
    
    # find FR in each bin
    firing_rate = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    spike_sem = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    for i in range(unique_bdx.shape[0]):
        spike_ct = np.sum(Y[b_idx == unique_bdx[i], :], axis=0)
        occupancy = dt * np.sum(b_idx==unique_bdx[i])
        spike_sem[i, :] = stats.sem(Y[b_idx == unique_bdx[i], :]/dt, axis=0)
        firing_rate[i, :] = spike_ct / occupancy
    if smooth:
        firing_rate = gaussian_filter1d(firing_rate, l, axis=0, mode='wrap')
        spike_sem = gaussian_filter1d(spike_sem, l, axis=0, mode='wrap')
    
    if SEM:
        return firing_rate, centers, spike_sem
    else:
        return firing_rate, centers


# population-wide analysis
def get_maps(Y, N=2, M=2):
    '''
    Use k-means clustering to identify different neural maps.
    Find N clusters, keep the trial index for the M most common maps (N >= M).

    Params:
    ------
    Y : ndarray
        firing rate for each cell; shape (n_trials, n_pos_bins, n_cells)
    N : int
        number of k-means clusters; optional, default is 2
    M : int
        number of maps to keep; optional, default is 2

    Returns:
    -------
    if M < N:
    W_new : bool
        True if neural activity occupied that map on that trial
        shape (n_trials, M)
        Columns are ordered by which map was most common.
    else:
    W : bool
        True if neural activity occupied that map on that trial
        shape (n_trials, N)
        Column/map order is arbitrary.
    H : tuning curve estimates; shape (n_maps, n_cells*n_pos_bins)
    '''
    # format the data
    n_trials = Y.shape[0]
    Y_flat = Y.reshape(n_trials, -1)

    # fit k-means model
    kmeans = KMeans(n_clusters=N, n_init=100, random_state=1234)
    kmeans.fit(Y_flat)
    H = kmeans.cluster_centers_
    W_raw = kmeans.labels_
    
    # reformat W to be (n_trials, n_maps)
    W = np.zeros([n_trials, N])
    for m in range(N):
        W[:, m] = (W_raw == m) 

    # keep trial index for the M most common maps
    if M < N:
        n_trials = np.sum(W, axis=0)
        W_new = np.zeros((W.shape[0], M))
        for m in range(M):
            m_idx = np.argmax(n_trials)
            W_new[:, m] = W[:, m_idx]
            n_trials[m_idx] = -1
        return W_new, H
    else:
        return W, H

def get_map_idx(W, trials):
    '''
    Get the k-means map index for each observation
    given the map index for each trial

    Params:
    ------
    W : bool
        True if neural activity occupied that map on that trial
        shape (n_trials, n_maps)
    trials : ndarray
        trial number for each observation; shape (n_obs,)

    Returns:
    -------
    map_idx : ndarray
        True if neural activity occupied that map on that observation
        shape (n_maps, n_obs)
    '''
    n_trials, n_maps = W.shape
    n_obs = trials.shape[0]
    map_idx = np.zeros([n_maps, n_obs])
    for i in range(n_trials):
        for j in range(n_maps):
            if W[i, j]:
                map_idx[j, trials == i] = 1
    return map_idx.astype(bool)

def spatial_similarity(Y):
    '''
    Calculate the trial-by-trial correlation in spatial representations
    across the neural population

    Params:
    ------
    Y : ndarray
        firing rate for each cell; shape (n_trials, n_pos_bins, n_cells)

    Returns:
    -------
    sim : ndarray
        correlation of spatial representations; shape (n_trials, n_trials)
    '''
    Y_unwrapped = np.reshape(Y, (Y.shape[0], -1))
    sim_vec = np.abs(dist.pdist(Y_unwrapped, 'correlation')-1)
    sim = dist.squareform(sim_vec)

    return sim

def clu_distance_population(Y, H, map_idx):
    '''
    Calculate the distance between the population activity and 
    the k-means cluster centroids on each trial

    Params:
    ------
    Y : ndarray
        normalized firing rate by 5cm position bins by trial for each cell
        shape (n_trials, n_pos_bins, n_cells)
    H : ndarray
        k-means tuning curve estimates for each cluster/map
        shape (n_maps, n_cell*n_pos_bins)
    map_idx : int
        index for map 1

    Returns:
    -------
    dist : ndarray
        distance to cluster on each trial; shape (n_trials, )
        1 = in map 1 centroid
        -1 = in map 2 centroid
        0 = exactly between the two maps
    '''
    # reshape Y to get a trial x neurons*positions matrix
    Y = Y.transpose(0, 2, 1)
    Y_unwrapped = np.reshape(Y, (Y.shape[0], -1))
    n_trials, n_cells, n_pos = Y.shape

    # get kmeans centroids
    c1 = H[map_idx]
    c2 = H[map_idx-1]
    
    # project everything down to a vector connecting the two centroids
    proj = (c1 - c2) / np.linalg.norm(c1 - c2)
    projc1 = c1 @ proj # cluster 1
    projc2 = c2 @ proj # cluster 2
    projY = Y_unwrapped @ proj # activity on each trial
    
    # get distance to cluster on each trial
    dd = (projY - projc2) / (projc1 - projc2)
    return 2 * (dd - .5) # classify -1 or 1