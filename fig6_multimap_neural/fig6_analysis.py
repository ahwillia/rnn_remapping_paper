import numpy as np
import sys
sys.path.append("../utils/")
sys.path.append("../fig1_1d2map/")

from scipy import stats
import itertools
from scipy.spatial.distance import pdist

import raw_neural_data as raw
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
    path = f'{data_folder}{session_ID}/'
    d = {}

    # FR by 5cm position bins by trial for each cell
    d['Y'] = np.load(f'{path}{session_ID}_MEC_FRtensor.npy')

    # spike count by observation for each cell
    d['B'] = np.load(f'{path}{session_ID}_MEC_spikes.npy')

    # behavioral variables by observation - position, speed, trial, time
    d['A'] = np.load(f'{path}{session_ID}_behavior.npy')

    # ID numbers for all good cells
    d['cells'] = np.load(f'{path}{session_ID}_MEC_cellIDs.npy')

    # Neuropixels data file
    d['sp'] = raw.load_np_data(data_folder, session_ID)
    
    # spike waveforms from select epochs of trials
    d['waveform_avg'] = np.load(f'{path}{session_ID}_mean_waveforms.npy')
    d['waveform_std'] = np.load(f'{path}{session_ID}_stddev_waveforms.npy')
    d['epochs'] = np.load(f'{path}{session_ID}_epochs.npy') # in samples
    d['epoch_trials'] = samples_to_trials(d) # in trials

    return d


def format_neural_data(d, n_maps=3,\
                        filter_stability=True,\
                        unstable_thresh=0.25):
    '''
    Performs k-means clustering to divide the session into maps
    Computes the network-wide trial-trial spatial similarity

    If filter_stability=True, filters out unstable trials
    using the threshold defined by unstable_thresh
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

    # get the trial-trial network similarity
    Y = d['Y'].copy()
    d['sim'] = spk.spatial_similarity(Y)

    if filter_stability:
        # filter the data to remove unstable trials
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
    epoch_trials = d['epoch_trials'].copy() # trials IDs defining waveform epochs
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
    d_filt = d.copy()
    d_filt['cells'] = cells
    d_filt['kmeans'] = {}

    # filter trial-based data
    trial_idx = np.sum(W_filt, axis=1).astype(bool)
    d_filt['Y'] = Y[trial_idx]
    d_filt['kmeans']['W'] = W[trial_idx]

    # filter epochs
    trial_nums = np.unique(trials)
    unstable_trials = trial_nums[(trial_idx - 1).astype(bool)]
    for t_start, t_stop in epoch_trials:
        while t_start in unstable_trials:
            t_start += 1
        while t_stop in unstable_trials:
            t_stop -= 1

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
        if t in epoch_trials:
            epoch_trials[epoch_trials==t] = i
    A_new[:, 2] = new_trials
    d_filt['A'] = A_new

    return d_filt


''' Geometry '''
def align_remap_dims(data_folder, session_IDs, num_maps):
    '''
    For each session, find the angle between the  remapping dimensions

    Returns
    -------
    all_angles : list of arrays
        the angles bewteen all remapping dimensions connected to the same node
    '''
    # data params
    n_sessions = len(session_IDs)
    n_maps = np.max(num_maps)
    n_pairs = (np.math.factorial(n_maps)) // \
                (np.math.factorial(2)*np.math.factorial(n_maps-2))
    dt = 0.02 # time bin
    pos_bin = 2 # cm
    n_pos_bins = 400 // pos_bin

    all_angles = []
    for i, s_id in enumerate(session_IDs):
        # define the map pairs
        n_maps = num_maps[i]
        m_ids = np.arange(n_maps)
        m_pairs = list(itertools.combinations(m_ids,2))

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
        
        # find the remapping dimensions
        remap_dims = {}
        for m0_id, m1_id in m_pairs:
            remap_dims[f'[{m0_id} {m1_id}]'] = remapping_dim(FRs[m0_id], FRs[m1_id])
        
        # calculate the angles between all dimensions sharing a node
        all_pairs = list(itertools.combinations(np.asarray(m_pairs),2))
        adj_pairs = []
        for j, k in all_pairs:
            m_diff = np.setdiff1d(j, k)
            if m_diff.shape[0] == 1:
                adj_pairs.append((str(j), str(k)))
        n_angles = len(adj_pairs)
        angles = np.zeros(n_angles)
        for j, (p0, p1) in enumerate(adj_pairs):
            angles[j] = cosine_sim(remap_dims[p0], \
                                    remap_dims[p1])
        all_angles.append(np.rad2deg(np.arccos(np.abs(angles))))
    
    return all_angles


''' Waveforms '''
def choose_epochs(data_folder, session_ID, n_maps, unstable_thresh=0.25):
    '''
    Define the epoch over which to extract waveforms for each session.
    Find the longest block of stable trials for each map.
    '''
    # load the data and divide by map
    d = load_neural_data(data_folder, session_ID)
    d = format_neural_data(d, n_maps=n_maps,
                                    filter_stability=False, unstable_thresh=0.25)
    sp = raw.load_np_data(data_folder, session_ID)
    d['sp'] = sp
    
    # data params
    W = d['kmeans']['W'].copy() # k-means cluster labels by trial
    sim = d['sim'].copy() # trial-trial similarity

    # get the indices for unstable trials for each map
    unstable_idx = []
    for j in range(n_maps):
        # extract the correlation matrix for this map
        m_idx = W[:, j].astype(bool)
        sim_map = sim[m_idx, :]
        sim_map = sim_map[:, m_idx]

        # compute the local stability for each trial
        n_trials_map = sim_map.shape[0]
        local_corr = np.zeros(n_trials_map)
        for t in range(n_trials_map):
            local_corr[t] = np.mean(sim_map[t])

        # store for each map
        unstable_idx.append(local_corr < unstable_thresh)

    # find blocks of trials for each map
    n_trials = W.shape[0]
    all_trials = np.arange(n_trials)
    epoch_trials = np.zeros((n_maps, 2))
    for j in range(n_maps):
        # find blocks of contiguous trials
        # u_idx = unstable_idx[j]
        trials = all_trials[W[:, j].astype(bool)]
        # trials_adj = trials.copy()
        # trials_adj[u_idx] = trials_adj[u_idx] * -100
        switches = np.diff(trials) > 1
        block_starts = trials[np.insert(switches, 0, True)]
        block_ends = trials[np.insert(switches, -1, True)]
        
        # use the first and last trial of the longest block
        block_len = block_ends - block_starts    
        epoch_trials[j, 0] = block_starts[np.argmax(block_len)]
        epoch_trials[j, 1] = block_ends[np.argmax(block_len)] 

    # get epochs in terms of samples
    d['epoch_trials'] = epoch_trials
    epoch_times = trials_to_samples(d)

    return epoch_trials, epoch_times



def samples_to_trials(d):
    '''
    Convert from epoch times (which are in samples) 
    to the corresponding trial numbers
    
    Params:
    -------
    epochs : ndarray, shape (num_epochs, 2) 
        sample numbers defining the start and end of each epoch
    '''
    # sampling params
    epoch_times = d['epochs']
    sample_rate = d['sp']['sample_rate']
    
    # data params
    timepoints = d['A'][:, -1]
    trials = d['A'][:, 2]
    n_epochs = epoch_times.shape[0]
    
    # get the trials for each epoch start/stop
    epoch_trials = np.zeros((n_epochs, 2))
    for epoch_idx, (epoch_start, epoch_stop) in enumerate(epoch_times):
        start_idx = np.argmin(np.abs(timepoints - epoch_start/sample_rate))
        stop_idx = np.argmin(np.abs(timepoints - epoch_stop/sample_rate))    
        epoch_trials[epoch_idx, 0] = trials[start_idx]
        epoch_trials[epoch_idx, 1] = trials[stop_idx]
    
    return epoch_trials

def trials_to_samples(d):
    '''
    Convert from epoch trials to epoch start/stop times
    in terms of kilosort samples
    
    Params:
    -------
    epoch_trials : ndarray, shape (num_epochs, 2) 
        trial numbers defining the start and end of each epoch
    '''
    # data params
    epoch_trials = d['epoch_trials']
    timepoints = d['A'][:, -1]
    trials = d['A'][:, 2]
    n_epochs = epoch_trials.shape[0]
    
    # sampling params
    sample_rate = d['sp']['sample_rate']
    
    # get the sample number for each epoch start/stop
    epoch_times = np.zeros((n_epochs, 2))
    for epoch_idx, (epoch_start, epoch_stop) in enumerate(epoch_trials):
        start_idx = np.argmin(np.abs(trials - epoch_start))
        stop_idx = np.argmin(np.abs(trials - epoch_stop))    
        epoch_times[epoch_idx, 0] = timepoints[start_idx] * sample_rate
        epoch_times[epoch_idx, 1] = timepoints[stop_idx] * sample_rate
    
    return epoch_times


def get_cell_channels(d):
    '''
    Uses each cell's recorded depth to find the 
    nearest probe channel to that cell

    Returns
    -------
    cell_channels : ndarray, shape (n_cells,)
        array of ints corresponding to the channel nearest to each cell
    '''
    # load relevant data
    cells = d['cells']
    sp = d['sp']
    depth_unsorted = sp['spike_depth'].copy()

    # get the indices for all "good" cells
    cgs = sp['cgs'].copy()
    cids = sp['cids'].copy()
    good_cells = cids[cgs == 2]

    # get depth for each MEC cell
    depth = np.zeros(cells.shape[0])
    for i, c in enumerate(cells):
        depth[i] = depth_unsorted[good_cells==c]

    # get nearest channel to each cell
    channel_depths = sp['ycoords'].copy()
    channels = np.zeros(cells.shape[0])
    for i, y in enumerate(depth):
        channels[i] = 3 + np.argmin(np.abs(channel_depths - y))
    
    return channels.astype(int)


def wf_correlations(d):
    '''
    Finds the Pearson correlation between the average waveforms
    from each map for all cells.

    Uses the 20 channels nearest to each cell.
    '''
    # load relevant data
    waveform_avg = d['waveform_avg'].copy()
    waveform_std = d['waveform_std'].copy()
    cell_channels = d['cell_channels'].copy()
    n_epochs, n_cells, n_channels, n_samples = waveform_avg.shape

    wf_avg = np.zeros((n_epochs, n_cells, 20, n_samples))
    for i, ch in enumerate(cell_channels):
        if ch < 10:
            wf_avg[:, i, :, :] = waveform_avg[:, i, :20, :]
        elif ch > 375:
            wf_avg[:, i, :, :] = waveform_avg[:, i, 365:, :]
        else:
            wf_avg[:, i, :, :] = waveform_avg[:, i, ch-10:ch+10, :]

    # compute correlation for flattened vectors
    avg_corr = np.zeros(n_cells)
    flat_wf_avg = np.reshape(wf_avg, (n_epochs, n_cells, -1))
    avg_corr = np.full(n_cells, np.nan)
    for c in range(n_cells):
        cell_wf_avg = flat_wf_avg[:, c, :]
        corr_vec = np.abs(pdist(cell_wf_avg, 'correlation')-1)
        avg_corr[c] = np.nanmean(corr_vec)

    # get quartiles and median
    pct_5 = np.percentile(avg_corr, 5)
    pct_95 = np.percentile(avg_corr, 95)
    med_corr = np.median(avg_corr)

    # print the results
    print(f'across all cells: median waveform correlation = {med_corr:.4}, 5th percentile = {pct_5:.4}')

    return avg_corr