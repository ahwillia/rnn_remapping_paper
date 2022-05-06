def tuning_curve_1d():
    '''
    ** IN PROGRESS **

    Get the binned firing rate for a given stimulus.

    Params
    ------
    x : nadarray
        variable of interest by observation; shape (n_obs,)
    y : ndarray

    '''
    n_units = X.shape[-1]
    n_pos_bins = 50
    bin_size = (2 * np.pi) / n_pos_bins

    # define the position bins
    edges = np.arange(-np.pi + bin_size, np.pi, bin_size)
    bin_idx = np.digitize(pos_targ, edges)

    # get binned firing rate by trial
    FR_by_traversal = np.zeros((n_traversals, n_units, n_pos_bins))
    for t in np.unique(trials_by_obs):
        trial_idx = (trials_by_obs == t)
        b_idx = bin_idx[trial_idx]
        trial_pos = pos_targ[trial_idx].copy()
        trial_fr = X[trial_idx].copy()
        
        for b in np.unique(b_idx):
            FR_by_traversal[t, :, b] = np.mean(trial_fr[b_idx==b], 
                                               axis=0)

    # smooth the firing rates over position
    FR_by_traversal = gaussian_filter1d(FR_by_traversal, 2, axis=-1, mode='wrap')

    # normalize the firing rates for each unit
    fr = FR_by_traversal.copy()
    fr -= np.min(fr, axis=(0, -1))[None, :, None]
    fr /= (np.max(fr, axis=(0, -1)) + 1e-9)[None, :, None]
    FR_by_traversal = fr.copy()