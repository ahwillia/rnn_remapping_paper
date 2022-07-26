import numpy as np
import torch
import sys
sys.path.append("../utils/")

from basic_analysis import tuning_curve_2d
from dim_alignment import position_subspace, remapping_dim, cosine_sim, proj_aB
from model_utils import load_model_params, sample_rnn_data, format_rnn_data

from sklearn.utils import check_random_state
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import gaussian_filter1d
from scipy import stats

''' position and remapping dims alignment to the inputs and outputs '''
def align_in_out(data_folder, model_IDs):
    n_models = len(model_IDs)

    # to store the projections
    remap_dim_angles = {
        "ctxt_in": np.zeros((n_models, 2)),
        "ctxt_out": np.zeros((n_models, 2)),
        "pos_in": np.zeros((n_models, 2)), 
        "pos_out": np.zeros((n_models, 4))
    }
    pos_dim_angles = {
        "ctxt_in": np.zeros((n_models, 2)),
        "ctxt_out": np.zeros((n_models, 2)),
        "pos_in": np.zeros((n_models, 2)), 
        "pos_out": np.zeros((n_models, 4))
    }

    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        model, _, _ = load_model_params(data_folder, m_id)
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targets, pos_targets = format_rnn_data(outputs["hidden_states"], \
                                                targets["map_targets"], \
                                                targets["pos_targets"])
        n_units = X.shape[-1]

        # split by context
        X0 = X[map_targets==0]
        X1 = X[map_targets==1]
        pos0 = pos_targets[map_targets==0]
        pos1 = pos_targets[map_targets==1]

        # find the remapping dimension
        remap_dim = remapping_dim(X0, X1)
        
        # position-binned firing rates (n_pos_bins, n_units)
        X0_binned = tuning_curve_2d(X0, pos0[:, 0], pos0[:, 1])
        X1_binned = tuning_curve_2d(X1, pos1[:, 0], pos1[:, 1])
        X0_flat = X0_binned.reshape(-1, n_units)
        X1_flat = X1_binned.reshape(-1, n_units)
        X_tc = np.stack((X0_flat, X1_flat))

        # find the position subspace
        pos_subspace = position_subspace(X_tc, num_pcs=3)
        
        # inputs
        ctxt_inp_w = model.linear_ih.weight[:, 2:]
        pos_inp_w = model.linear_ih.weight[:, :2]
        ctxt_inp_w = ctxt_inp_w.detach().numpy() # (hidden_size, n_maps)
        pos_inp_w = pos_inp_w.detach().numpy() # (hidden_size, n_pos_dim)

        # outputs
        ctxt_out_w = model.readout_layer_map.weight
        ctxt_out_w = ctxt_out_w.detach().numpy().T # (hidden_size, n_maps)
        pos_out_w = model.readout_layer_pos.weight
        pos_out_w = pos_out_w.detach().numpy().T # (hidden_size, n_pos_dim * 2)
        
        # find the alignments
        all_weights = [ctxt_inp_w, ctxt_out_w, pos_inp_w, pos_out_w]
        for label, w in zip(remap_dim_angles.keys(), all_weights):
            remap_dim_angles[label][i] = np.abs(cosine_sim(remap_dim, w))
            pos_dim_angles[label][i] = np.abs(proj_aB(w, pos_subspace))

    return remap_dim_angles, pos_dim_angles


''' to get a 1D slice of the 2D manifold '''
def get_slice_data(data_folder, model_ID, fix_x=True):
    '''
    Generates RNN data that slices through the 2D torus

    Returns
    -------
    X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    pos_targ : ndarray, shape (n_obs,)
        track positions at each observation (rad)
    map_targ : ndarray of ints, shape (n_obs,)
        context at each observation
    '''
    # set random seeds
    NP_SEED = int(model_ID.split('_')[0])
    TORCH_SEED = int(model_ID.split('_')[1])
    random_state = np.random.RandomState(NP_SEED)
    torch.manual_seed(TORCH_SEED)

    # set params
    model, task_params, rnn_params = load_model_params(data_folder, model_ID)
    n_batch = 50
    session_params = {
        'fix_x': fix_x,
        'num_maps': 2,
        'num_steps': 330,
        'remap_pulse_duration': 2,
        'remap_rate': 0.002,
        'velocity_drift_stddev': 0.1,
        'velocity_noise_stddev': 0.3,
        'num_spatial_dimensions': 2
        }

    # get sample neural activity, position targets, map targets, inputs
    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
            generate_batch_slice(n_batch, random_state, **session_params)
    pos_outputs, map_logits, hidden_states = model(inp_init, inp_vel, inp_remaps)
    X, map_targ, pos_targ = format_rnn_data(hidden_states, map_targets, pos_targets)

    return X, pos_targ, map_targ


def generate_batch_slice(
        batch_size,
        random_state,
        **kwargs
    ):
    """
    Generates a batch of trials, 

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
        init, vel, remaps, pos, mp = generate_trial_slice(rs, **kwargs)

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
    


def generate_trial_slice(
        random_state,
        fix_x=True,
        num_steps=300,
        num_maps=2,
        remap_rate=0.02,
        velocity_drift_stddev=0.01,
        velocity_noise_stddev=0.03,
        remap_pulse_duration=5,
        num_spatial_dimensions=1,
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
    possible_remap_times = np.arange(remap_pulse_duration, num_steps, remap_pulse_duration)
    num_remaps = np.clip(rs.poisson(num_steps * remap_rate), 0, len(possible_remap_times))

    # Sample initial position
    initial_position = rs.uniform(low=-np.pi, high=np.pi, size=num_spatial_dimensions)
    if fix_x:
        initial_position[0] = 0
    else:
        initial_position[1] = 0
    inp_init = np.array([[np.cos(p), np.sin(p)] for p in initial_position]).ravel()

    # Sample velocity trace.
    v_mean = velocity_drift_stddev * rs.randn(1, num_spatial_dimensions)
    v_noise = velocity_noise_stddev * rs.randn(num_steps, num_spatial_dimensions)
    inp_vel = v_mean + v_noise
    if fix_x:
        inp_vel[:, 0] = 0
    else:
        inp_vel[:, 1] = 0

    # Compute position by integrating velocity.
    pos_targets = initial_position + np.cumsum(inp_vel, axis=0)

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