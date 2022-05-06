import numpy as np


def tuning_curve_1d(fr, pos, n_bins=80,\
          pos_min=-np.pi, pos_max=np.pi, return_pos=False):

    '''
    Compute a 1D tuning curve.
    
    Params
    ------
    fr : ndarray, shape (n_obs, n_units)
        firing rate of each cell at each observation
    pos : ndarray, shape (n_obs, )
        corresponding positions
    n_bins : int, num pos bins
    pos_min, pos_max : to specify pos bins
    return_pos : bool
        if True, also returns the position bin centers
    
    Returns
    -------
    tc : ndarray, shape (n_bins, n_units)
        avg firing rate in each position bin
    pos_centers : ndarray (n_bins, )
        center of each position bin
    '''
    pos_bins = np.linspace(pos_min, pos_max, n_bins+1)
    n_units = fr.shape[1]

    b_idx = np.digitize(pos, pos_bins)
    unique_bins = np.unique(b_idx)
    tc = np.zeros((n_bins, n_units))
    for i, b in enumerate(unique_bins):
        tc[i] = np.mean(fr[b_idx==b], axis=0)

    if return_pos:
        pos_centers = np.mean((pos_bins[:-1], pos_bins[1:]), axis=0)
        return tc, pos_centers
    else:
        return tc


def tuning_curve_2d(fr, x_pos, y_pos, n_bins=80, \
          pos_min=-np.pi, pos_max=np.pi, return_pos=False):
    '''
    Compute a 2D tuning curve.
    Assumes a square environment (xdims = ydims)

    Based on ln-model-of-mec-neurons/compute_2d_tuning_curve.m by Kiah Hardcastle
    
    Params
    ------
    fr : ndarray, shape (n_obs, n_units)
        firing rate of each cell at each observation
    x_pos : ndarray, shape (n_obs, )
        x-axis positions
    y_pos : ndarray, shape (n_obs, )
        y-axis positions
    n_bins : int, num pos bins
    pos_min, pos_max : to specify pos bins
    return_pos : bool
        if True, also returns the position bin centers
    
    Returns
    -------
    tc : ndarray, shape (y_bins, x_bins, n_units)
        avg firing rate in each xy position bin
    x_centers, y_centers : ndarray (n_bins, n_bins)
        center of each position bin
        organized in terms of x or y
    '''
    pos_bins = np.linspace(pos_min, pos_max, n_bins+1)   
    n_units = fr.shape[1]
    
    # fill in the average firing rates for each pos bin
    tc = np.zeros((n_bins, n_bins, n_units))
    # in y bin
    for i in range(n_bins):
        y_start = pos_bins[i]
        y_stop = pos_bins[i+1]
        if i == n_bins:
            y_idx = (y_pos >= y_start) & (y_pos <= y_stop)
        else:
            y_idx = (y_pos >= y_start) & (y_pos < y_stop)

        # in x bin
        for j in range(n_bins):
            x_start = pos_bins[j]
            x_stop = pos_bins[j+1]
            if j == n_bins:
                x_idx = (x_pos >= x_start) & (x_pos <= x_stop)
            else:
                x_idx = (x_pos >= x_start) & (x_pos < x_stop)
                
            tc[i, j] = np.mean(fr[x_idx & y_idx], axis=0)
                
    # fill in nans with neighboring values
    for n in range(n_units):
        nan_idx = np.isnan(tc[:, :, n])
        [i_idx, j_idx] = np.where(nan_idx)  
        for i, j in zip(i_idx, j_idx):
            try:
                right = tc[i, j+1]
            except:
                right = tc[i, 0]
            try:
                left = tc[i, j-1]
            except:
                left = tc[i, n_bins-1]
            try:
                down = tc[i-1, j]
            except:
                down = tc[n_bins-1, j]
            try:
                up = tc[i+1, j]
            except:
                up = tc[0, j]
            tc[i, j] = np.nanmean((right, left, up, down))
            
    if return_pos:
        pos_avg = np.mean((pos_bins[:-1], pos_bins[1:]), axis=0)
        x_centers, y_centers = np.meshgrid(pos_avg, pos_avg)
        return tc, x_centers, y_centers
    else:
        return tc


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
    rmse_shuff_mean : float
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
    rmse_shuff_mean = np.percentile(rmse_2, 2.5)
    
    # 5) Compute the relative misalignment score
    norm_align = (rmse_raw - rmse_aligned) / (rmse_shuff_mean - rmse_aligned)
    
    return norm_align, rmse_raw, rmse_aligned, rmse_shuff_mean