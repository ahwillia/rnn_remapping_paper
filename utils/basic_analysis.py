import numpy as np
from scipy.ndimage import gaussian_filter1d

def tuning_curve_1d(X, pos,\
                    n_pos_bins=50, pos_min=-np.pi, pos_max=np.pi,\
                    smooth=False, normalize=False):
    '''
    Get the position-binned firing rates.

    Params
    ------
    X : ndarray
        firing rates; shape (n_obs, n_units)
    pos : nadarray
        positions; shape (n_obs,)

    Returns
    -------
    tc : ndarray, shape (n_pos_bins, n_units)
        avg firing rate in each position bin
    binned_pos : ndarray (n_pos_bins, )
        center of each position bin
    '''
    n_obs, n_units = X.shape

    # define the position bins
    bin_size = (pos_max - pos_min) / n_pos_bins
    edges = np.arange(pos_min + bin_size, pos_max, bin_size)
    bin_idx = np.digitize(pos, edges)

    # get binned positions
    binned_pos = np.linspace(pos_min, pos_max, num=n_pos_bins)

    # get binned firing rates
    tc = np.zeros((n_pos_bins, n_units))        
    for b in np.unique(bin_idx):
        tc[b, :] = np.mean(X[bin_idx==b], axis=0)

    # smooth the firing rates over position
    if smooth:
        tc = gaussian_filter1d(tc, 2, axis=0, mode='wrap')

    # normalize the firing rates for each unit
    if normalize:
        tc -= np.min(tc, axis=0)[None, :]
        tc /= (np.max(tc, axis=0) + 1e-9)[None, :]

    return tc, binned_pos


def tuning_curve_2d(X, x_pos, y_pos, \
                    n_pos_bins=80, pos_min=-np.pi, pos_max=np.pi, \
                    smooth=False, return_pos=False):
    '''
    Compute a 2D tuning curve.
    Assumes a square environment (xdims = ydims)

    Based on ln-model-of-mec-neurons/compute_2d_tuning_curve.m by Kiah Hardcastle
    
    Params
    ------
    X : ndarray, shape (n_obs, n_units)
        firing rate of each cell at each observation
    x_pos : ndarray, shape (n_obs, )
        x-axis positions
    y_pos : ndarray, shape (n_obs, )
        y-axis positions
    
    Returns
    -------
    tc : ndarray, shape (y_bins, x_bins, n_units)
        avg firing rate in each xy position bin
    x_centers, y_centers : ndarray (n_pos_bins, n_pos_bins)
        center of each position bin
        organized in terms of x or y
    '''
    pos_bins = np.linspace(pos_min, pos_max, n_pos_bins+1)   
    n_units = X.shape[1]
    
    # fill in the average firing rates for each pos bin
    tc = np.zeros((n_pos_bins, n_pos_bins, n_units))
    # in y bin
    for i in range(n_pos_bins):
        y_start = pos_bins[i]
        y_stop = pos_bins[i+1]
        if i == n_pos_bins:
            y_idx = (y_pos >= y_start) & (y_pos <= y_stop)
        else:
            y_idx = (y_pos >= y_start) & (y_pos < y_stop)

        # in x bin
        for j in range(n_pos_bins):
            x_start = pos_bins[j]
            x_stop = pos_bins[j+1]
            if j == n_pos_bins:
                x_idx = (x_pos >= x_start) & (x_pos <= x_stop)
            else:
                x_idx = (x_pos >= x_start) & (x_pos < x_stop)
                
            tc[i, j] = np.mean(X[x_idx & y_idx], axis=0)
                
    # fill in nans with neighboring values
    for n in range(n_units):
        nan_idx = np.isnan(tc[:, :, n])
        [i_idx, j_idx] = np.where(nan_idx)  
        for i, j in zip(i_idx, j_idx):
            try:
                right = tc[i, j+1]
            except:
                right = tc[i, 0]
            left = tc[i, j-1]
            down = tc[i-1, j]
            try:
                up = tc[i+1, j]
            except:
                up = tc[0, j]
            tc[i, j] = np.nanmean((right, left, up, down))
    
    # smooth the firing rates over position
    if smooth:
        tc = gaussian_filter1d(tc, 2, axis=0, mode='wrap')
        tc = gaussian_filter1d(tc, 2, axis=1, mode='wrap')

    if return_pos:
        pos_avg = np.mean((pos_bins[:-1], pos_bins[1:]), axis=0)
        x_centers, y_centers = np.meshgrid(pos_avg, pos_avg)
        return tc, x_centers, y_centers
    else:
        return tc


''' MANIFOLD GEOMETRY AND ALIGNMENT '''
def compute_misalignment(x0, x1):
    '''
    Compute the misalignment of two manifolds, specified by x0 and x1,
    normalized to the best rotational alignment (0) and alignment
    after random rotations of one manifold (1).

    Params
    ------
    x0, x1 : ndarray, shape (n_stim, n_units)
        stimuli (e.g. pos bins) and units should be matched

    Returns
    -------
    norm_align : float
        normalized misalignment score
    rmse_raw : float
        raw rmse of the two normalized and mean-centered manifolds
    rmse_aligned : float
        rmse after the best rotational alignment
    rmse_shuff_thresh : float
        2.5 percentile of shuffled alignment (after random rotation)
    '''

    # 1) Mean-center cluster centroids.
    m1 = x0 - np.mean(x0, axis=0, keepdims=True)
    m2 = x1 - np.mean(x1, axis=0, keepdims=True)

    m1_norm = np.linalg.norm(m1)
    m2_norm = np.linalg.norm(m2)

    m1 /= m1_norm
    m2 /= m2_norm

    # 2) Compute Raw RMSE
    rmse_raw = np.sqrt(np.mean((m1 - m2) ** 2))

    # 3) Compute RMSE after best rotational alignment
    u, _, vt = np.linalg.svd(m1.T @ m2)
    rmse_aligned = np.sqrt(np.mean((m1 @ (u @ vt) - m2) ** 2))

    # 4) Compute Null Distribution of RMSE's by random rotations
    rmse_2 = []
    for _ in range(100):
        Q = np.linalg.qr(np.random.randn(m1.shape[1], m1.shape[1]))[0]
        rmse_2.append(np.sqrt(np.mean((m1 @ Q - m2) ** 2)))
    rmse_shuff_thresh = np.percentile(rmse_2, 2.5)
    
    # 5) Compute the relative misalignment score
    norm_align = (rmse_raw - rmse_aligned) / (rmse_shuff_thresh - rmse_aligned)
    
    return norm_align, rmse_raw, rmse_aligned, rmse_shuff_thresh