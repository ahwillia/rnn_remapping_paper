import numpy as np
import torch

from sklearn.utils import check_random_state
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import gaussian_filter1d

''' to create RNN data analogous to biological data '''
def generate_batch_pos_vel(
        batch_size,
        random_state,
        **kwargs
    ):
    """
    Generates a batch of trials, Forward velocity only

    Parameters
    ----------
    batch_size : int, number of trials to generate
    random_state : int, seed for random numbers
    **kwargs : passed to generate_trial function

    Returns
    -------
    inp_init : torch.tensor, shape == (batch_size, 2)
    inp_remaps : torch.tensor, shape == (num_steps, batch_size, num_maps)
    inp_vel : torch.tensor, shape == (num_steps, batch_size)
    pos_targets : torch.tensor, shape == (num_steps, batch_size)
    map_targets : torch.tensor of ints, shape == (num_steps, batch_size)
    """
    
    inp_init = []
    inp_remaps = []
    inp_vel = []
    pos_targets = []
    map_targets = []
    rs = check_random_state(random_state)

    for k in range(batch_size):
        init, vel, remaps, pos, mp = generate_trial_pos_vel(rs, **kwargs)

        inp_init.append(init)
        inp_remaps.append(remaps)
        inp_vel.append(vel)
        pos_targets.append(pos)
        map_targets.append(mp)


    # (batch_size, 2)
    inp_init = torch.tensor(
        np.stack(inp_init), dtype=torch.float32
    )

    # (num_steps, batch_size, num_maps)
    inp_remaps = torch.tensor(
        np.stack(inp_remaps, axis=1), dtype=torch.float32
    )

    # (num_steps, batch_size, 1)
    inp_vel = torch.tensor(
        np.stack(inp_vel, axis=1), dtype=torch.float32
    )
    
    # (num_steps, batch_size, 2)
    pos_targets = torch.tensor(
        np.stack(pos_targets, axis=1), dtype=torch.float32
    )

    # (num_steps, batch_size)
    map_targets = torch.tensor(
        np.stack(map_targets, axis=1), dtype=torch.long
    )

    return inp_init, inp_remaps, inp_vel, pos_targets, map_targets
    

def generate_trial_pos_vel(
        random_state,
        num_steps=300,
        num_maps=2,
        remap_rate=0.02,
        velocity_drift_stddev=0.01,
        velocity_noise_stddev=0.03,
        remap_pulse_duration=5,
    ):
    """
    Parameters
    ----------
    random_state : int, seed for random numbers.
    num_steps : int, trial length.
    remap_rate : float, positive number controlling number of remap events.
    velocity_drift_stddev : float, standard deviation of average velocity.
    velocity_noise_stddev : float, controls noise in the velocity.
    remap_pulse_duration : int, how long a remap signal lasts.

    Returns
    -------
    inp_init : length-2 vector, represents initial position on ring.
    inp_vel : vector, len == num_steps, velocity at each time step.
    inp_remaps : matrix, shape (num_steps, num_maps), one-hot remap inputs.
    pos_targets : vector, len == num_steps, target position in radians.
    map_targets : vector of ints, len == num_steps, which map the network is in.
    """

    # Seed random number generator.
    rs = check_random_state(random_state)

    # Sample number of remaps.
    possible_remap_times = np.arange(remap_pulse_duration,\
                                        num_steps, remap_pulse_duration)
    num_remaps = np.clip(rs.poisson(num_steps * remap_rate), \
                                        0, len(possible_remap_times))

    # Set initial position to track start
    initial_position = -np.pi
    inp_init = np.array([
        np.cos(initial_position),
        np.sin(initial_position)
    ])

    # Sample velocity trace.
    v_mean = velocity_drift_stddev * rs.randn()
    v_noise = velocity_noise_stddev * rs.randn(num_steps)
    inp_vel = np.abs(v_mean + v_noise)[:, None] # force velocity to be non-negative
    inp_vel[0] = 0 # force starting position to be track start

    # Compute position by integrating velocity.
    pos_targets = (initial_position + np.cumsum(inp_vel))[:, None]

    # Generate sequence of map ids. Choose first map randomly.
    map_ids = np.zeros(num_remaps + 1, dtype='int32')
    map_ids[0] = rs.choice(np.arange(num_maps))
    for i in range(1, num_remaps + 1):

        # Must remap to a new / unique map.
        avail_maps = np.setdiff1d(np.arange(num_maps), map_ids[i - 1])

        # Pick new map randomly
        map_ids[i] = rs.choice(avail_maps)

    # Generate random remap times.
    remap_times = np.sort(
        rs.choice(
            possible_remap_times,
            replace=False,
            size=num_remaps
        )
    )
    remap_times = np.concatenate(([0], remap_times))

    # Construct map decoding targets.
    map_dwell_times = np.diff(np.concatenate((
        remap_times, [num_steps]
    )))
    map_targets = []
    for m, L in zip(map_ids, map_dwell_times):
        map_targets.append(np.full(L, m))
    map_targets = np.concatenate(map_targets)

    # Construct remapping input pulses
    inp_remaps = np.zeros((num_steps, num_maps))
    for t, m in zip(remap_times, map_ids):
        inp_remaps[t, m] = 1.0

    inp_remaps = maximum_filter1d(
        inp_remaps, remap_pulse_duration, axis=0
    )

    return inp_init, inp_vel, inp_remaps, pos_targets, map_targets


''' to work with RNN data analogous to biological data '''
def format_data(X_raw, pos_targ_raw, map_targ_raw):
    '''
    Remove the steps from the unfinished track traversal from each trial end
    Concatenate all trials together into a flattened array

    n_obs = num_steps*batch_size - num_unfinished_steps

    Params
    ------
    X_raw : ndarray, shape (num_steps, batch_size, hidden_size)
        RNN unit activity at each observation
    pos_targ_raw : ndarray, shape (num_steps, batch_size)
        track positions at each observation (rad)
    map_targ_raw : ndarray of ints, shape (num_steps, batch_size)
        context at each observation

    Returns
    ------
    new_X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    new_pos_targ : ndarray, shape (n_obs,)
        track positions at each observation (rad)
    new_map_targ : ndarray of ints, shape (n_obs,)
        context at each observation
    '''
    n_steps, n_batch, n_units = X_raw.shape

    new_map_targ = np.asarray([])
    new_pos_targ = np.asarray([])
    for i in range(n_batch):    
        # find the indices for each traversal start    
        pos = np.squeeze(pos_targ_raw[:, i])
        traversal_bool = np.abs(np.diff(pos)) > 5
        traversal_bool = np.insert(traversal_bool, 0, True)
        
        # trim off the unfinished traversal
        traversal_idx = np.where(traversal_bool)[0]
        if i == 0:
            new_X = X_raw[:traversal_idx[-1], i]
        else:
            new_X = np.concatenate((new_X,\
                                    X_raw[:traversal_idx[-1], i]))
        new_map_targ = np.append(new_map_targ,\
                                    map_targ_raw[:traversal_idx[-1], i])
        new_pos_targ = np.append(new_pos_targ,\
                                    pos[:traversal_idx[-1]])

    return new_X, new_pos_targ, new_map_targ


def traversal_num(pos_targ):
    '''
    Params
    ------
    pos_targ : ndarray, shape (n_obs,)
        track positions at each observation (rad)

    Returns
    -------
    traversals_by_obs : ndarray, shape (n_obs,)
        traversal number at each observation (starts at 0)
    '''
    traversal_starts = np.abs(np.diff(pos_targ)) > 5
    traversal_starts = np.insert(traversal_starts, 0, True)
    return np.cumsum(traversal_starts) - 1


def map_by_traversal(map_targets, traversals_by_obs):
    '''
    Label each traversal by the predominant context on that traversal

    Params
    ------
    map_targets : ndarray of ints, shape (n_obs,)
        context at each observation
    traversals_by_obs : ndarray, shape (n_obs,)
        traversal number at each observation (starts at 0)

    Returns
    -------
    map_idx : ndarray of ints, shape (n_traversals, n_maps)
        predominant context on each traversal
    '''
    n_traversals = np.max(traversals_by_obs) + 1
    trial_map_raw = np.zeros(n_traversals)
    for t in np.unique(traversals_by_obs):
        trial_map_raw[t] = np.mean(map_targets[traversals_by_obs==t])
    trial_map = np.round(trial_map_raw).astype(int)
    map_idx = np.stack((trial_map, np.abs(trial_map - 1)), axis = -1)

    return map_idx


def fr_by_traversal(X, pos_targ, traversals_by_obs,\
                    n_pos_bins=50, pos_min=-np.pi, pos_max=np.pi):
    '''
    Get the position-binned firing rate for each unit on each traversal
    Smoothes over position and normalizes within each unit

    Params
    ------
    X : ndarray, shape (n_obs, hidden_size)
        firing rates at each observation
    pos_targ : ndarray, shape (n_obs,)
        track positions at each observation (rad)
    traversals_by_obs : ndarray, shape (n_obs,)
        traversal number at each observation (starts at 0)

    Return
    ------
    FR_by_traversal : ndarray, shape (n_traversals, n_units, n_pos_bins)
        position-binned firing rates for each track traversal
    '''
    n_obs, n_units = X.shape
    n_traversals = np.max(traversals_by_obs) + 1

    # define the position bins
    bin_size = (pos_max - pos_min) / n_pos_bins
    edges = np.arange(pos_min + bin_size, pos_max, bin_size)
    bin_idx = np.digitize(pos_targ, edges)

    # get binned firing rate by trial
    FR_by_traversal = np.zeros((n_traversals, n_units, n_pos_bins))
    for t in np.unique(traversals_by_obs):
        trial_idx = (traversals_by_obs == t)
        b_idx = bin_idx[trial_idx]
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
    FR_by_traversal = fr

    return FR_by_traversal