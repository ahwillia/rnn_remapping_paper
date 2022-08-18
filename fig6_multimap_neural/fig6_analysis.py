import numpy as np
import sys
sys.path.append("../utils/")
sys.path.append("../fig1_1d2map/")

from scipy import stats

from basic_analysis import tuning_curve_1d
from dim_alignment import position_subspace, remapping_dim, cosine_sim, proj_aB
import analysis_neuro as spk

''' data munging '''
def load_neural_data(data_folder, session_ID):
    '''
    Loads neural data and packages in a dict

    session_ID : string
        mouse_session
    '''
    d = {}

    # FR by 5cm position bins by trial for each cell
    d['Y'] = np.load(f'{data_folder}{session_ID}_MEC_FRtensor.npy')

    # spike count by observation for each cell
    d['B'] = np.load(f'{data_folder}{session_ID}_MEC_spikes.npy')

    # behavioral variables by observation - position, speed, trial, time
    d['A'] = np.load(f'{data_folder}{session_ID}_behavior.npy')

    # ID numbers for all good cells
    d['cells'] = np.load(f'{data_folder}{session_ID}_MEC_cellIDs.npy')

    return d

def format_neural_data(d, n_maps=3, filter_stability=True, unstable_thresh=0.25):
    '''
    Performs k-means clustering to divide the session into maps
    Computes the network-wide trial-trial spatial similarity

    If filter_stability=True, filters out unstable trials
    '''
    # use k-means to get the map label for each trial
    Y = d['Y'].copy()
    W, _ = spk.get_maps(Y, N=n_maps, M=n_maps)
    d['kmeans'] = {}
    d['kmeans']['W'] = W

    # get observation idx for each map
    trials = d['A'][:, 2].copy()
    map_idx = spk.get_map_idx(W, trials)
    d['idx'] = map_idx

    if filter_stability:
        # filter the data to remove unstable trials
        Y = d['Y'].copy()
        d['sim'] = spk.spatial_similarity(Y)
        d = filter_by_stability(d, unstable_thresh=unstable_thresh)

        # recompute the observation idx for each map
        trials = d['A'][:, 2].copy()
        W = d['kmeans']['W'].copy()
        map_idx = spk.get_map_idx(W, trials)
        d['idx'] = map_idx

        # recompute the trial-trial network similarity
        Y = d['Y'].copy()
        d['sim'] = spk.spatial_similarity(Y)

    return d

def filter_by_stability(d, unstable_thresh=0.25):
    '''
    Filters the data to remove unstable trials whose network-wide spatial firing
    is not well-correlated with most other trials within the same map.

    Cut-off for the mean correlation is given by unstable_thresh.
    '''
    # load relevant data
    A = d['A'].copy() # behavior
    B = d['B'].copy() # spikes
    Y = d['Y'].copy() # firing rates
    cells = d['cells'].copy() # cell IDs
    sim = d['sim'].copy() # trial-trial similarity
    W = d['kmeans']['W'].copy() # k-means cluster labels by trial
    W = W.astype(bool)
    map_idx = d['idx'].copy() # k-means cluster labels by observation

    # data params
    n_trials, n_pos_bins, n_cells = Y.shape
    n_obs = A.shape[0]
    n_maps = W.shape[1]

    # get the indices for unstable trials
    unstable_idx = []
    for j in range(n_maps):
        # extract the correlation matrix for this map
        m_idx = W[:, j]
        sim_map = sim[m_idx, :]
        sim_map = sim_map[:, m_idx]
        
        # compute the local stability for each trial
        n_trials = sim_map.shape[0]
        local_corr = np.zeros(n_trials)
        for t in range(n_trials):
            local_corr[t] = np.mean(sim_map[t])
        
        # store for each map
        unstable_idx.append(local_corr < unstable_thresh)

    # get the indices for unstable observations
    trials = A[:, 2]
    unstable_obs_idx = np.zeros(n_obs)
    for j in range(n_maps):
        u_idx = unstable_idx[j]
        map_trials = trials[map_idx[j]].copy()
        for i, t in enumerate(np.unique(map_trials)):
            if u_idx[i]:
                unstable_obs_idx[trials==t] = True
    unstable_obs_idx = unstable_obs_idx.astype(bool)

    # filter W
    W_filt = W.copy()
    for j in range(n_maps):
        u_idx = unstable_idx[j].astype(bool)
        if np.sum(u_idx):
            map_trials = np.where(W[:, j])[0]
            for i, t in enumerate(map_trials):
                if u_idx[i]:
                    W_filt[t, j] = False

    # to store new data
    d_filt = {}
    d_filt['cells'] = cells
    d_filt['kmeans'] = {}


    # filter observation-based data
    A_new = A[~unstable_obs_idx]
    d_filt['B'] = B[~unstable_obs_idx]
    d_filt['idx'] = map_idx[:, ~unstable_obs_idx]

    # relabel trials
    trials = A_new[:, 2]
    n_obs = trials.shape[0]
    new_trials = np.zeros(n_obs)
    for i, t in enumerate(np.unique(trials)):
        new_trials[trials==t] = i
    A_new[:, 2] = new_trials
    d_filt['A'] = A_new

    # filter trial-based data
    trial_idx = np.sum(W_filt, axis=1).astype(bool)
    d_filt['Y'] = Y[trial_idx]
    d_filt['kmeans']['W'] = W[trial_idx]

    return d_filt


''' Geometry '''
def align_remap_dims(data_folder, session_IDs, num_maps):
    '''
    For each session, find the angle between the  remapping dimensions
    '''
    # data params
    dt = 0.02 # time bin
    pos_bin = 2 # cm
    n_pos_bins = 400 // pos_bin

    avg_angle = np.zeros(len(session_IDs))
    for i, s_id in enumerate(session_IDs):
        n_maps = num_maps[i]

        # load + format the data
        d = load_neural_data(data_folder, s_id)
        d = format_neural_data(d, n_maps=n_maps)

        # compute firing rates for each map
        A = d['A']
        B = d['B']
        n_cells = B.shape[-1]
        map_idx = d['idx']
        FRs = np.zeros([n_maps, n_pos_bins, n_cells])
        for j in range(n_maps):
            m_idx = map_idx[j, :]
            FRs[j], _, _ = spk.tuning_curve(A[m_idx, 0],
                                            B[m_idx, :],
                                            dt, b=2, SEM=True)
        
        remap_dims = []
        for j in range(n_maps):
            # pairwise comparison of maps
            m0_id = j
            m1_id = (j+1)%n_maps

            # find the remapping dimension
            remap_dims.append(remapping_dim(FRs[m0_id], FRs[m1_id]))
        
        angles = np.zeros(n_maps)
        for j in range(n_maps):
            angles[j] = cosine_sim(remap_dims[j], \
                                    remap_dims[(j+1)%n_maps])
        avg_angle[i] = np.rad2deg(np.arccos(np.abs(np.mean(angles))))
    
    session_avg = int(np.mean(avg_angle))
    session_sem = stats.sem(avg_angle)
    print(f'angle between remapping dims (mean, sem) = {session_avg} deg., {session_sem:.2} deg.')