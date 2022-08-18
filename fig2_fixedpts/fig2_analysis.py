import numpy as np
import torch
import sys
sys.path.append("../utils/")

from scipy import stats
from scipy.linalg import subspace_angles
from tqdm import trange

from basic_analysis import tuning_curve_1d
from dim_alignment import position_subspace, remapping_dim, cosine_sim, proj_aB
from model_utils import load_model_params, sample_rnn_data, format_rnn_data

''' Load and format the data '''
def load_fixed_pts(data_folder, model_ID, **kwargs):
    '''
    Loads and formats the location in neural activity space
    and the predicted position for each of the fixed points for 
    a given model.

    Returns
    -------
    fixed_pts : ndarray, shape (num_points, hidden_size)
        locations in activity space for each fixed point
    fixed_pts_torch : torch array, shape (num_points, hidden_size)
        locations in activity space for each fixed point
    pos_pred_fp : ndarray, shape (num_points,)
        predicted track position for each fixed point
    '''
    model, task_params, _ = load_model_params(data_folder, model_ID)

    # load the fixed points
    fixed_pts = np.load(f"{data_folder}{model_ID}/states_fixed_pt.npy")
    fixed_pts_torch = torch.from_numpy(fixed_pts)
    num_pts = fixed_pts.shape[0]

    # load the predicted positions
    pos_pred_fp = np.load(f"{data_folder}{model_ID}/states_fixed_pt.npy")
    pos_pred_fp = np.arctan2(pos_pred_fp[:, 1], pos_pred_fp[:, 0])

    # filter out points where the velocity is too great
    vel_idx = filter_by_velocity(model, task_params, \
                                    fixed_pts_torch, \
                                    **kwargs)
    fixed_pts = fixed_pts[vel_idx]
    fixed_pts_torch = fixed_pts_torch[vel_idx]
    pos_pred_fp = pos_pred_fp[vel_idx]
    print(f'filtered out {num_pts - np.sum(vel_idx)} slow points with velocity > threshold')

    return fixed_pts, fixed_pts_torch, pos_pred_fp


def filter_by_velocity(model, task_params, \
                        fixed_pts_torch, \
                        vel_thresh=0.005):
    '''
    Finds index for slow points with velocity greater than vel_thresh.
    
    Returns
    -------
    vel_idx : bool, shape (num_pts)
        True if fixed point velocity is less than vel_thresh
    '''
    # data params
    num_pts = fixed_pts_torch.shape[0]
    num_pos_dims = 1
    # num_pos_dims = task_params["num_spatial_dimensions"]
    num_maps = task_params["num_maps"]

    # inputs
    inp_vel = torch.zeros(num_pts, num_pos_dims)
    inp_remaps = torch.zeros(num_pts, num_maps)

    # run the model for one step
    X1 = fixed_pts_torch
    X2 = model.one_step(fixed_pts_torch, \
                        inp_vel, inp_remaps)
    X1 = X1.detach().numpy()
    X2 = X2.detach().numpy()

    # calculate the velocity
    vels = np.asarray([])
    for x1, x2 in zip(X1, X2): 
        vel = np.sum((x1 - x2)**2)
        vels = np.append(vels, vel)
    vel_idx = vels < vel_thresh

    return vel_idx


def eig_vec_examples(stable_idx, saddle_idx, \
                        im_idx, fp_dist, \
                        n_ex = 16, map_thresh=0.1):
    '''
    Get the indices for example fixed points for panels F & G.

    Chooses n_ex example points from each manifold and the
    separatrix for a total of roughly 3 * n_ex example points
    (minus positions for which there is no nearby fixed point
    that meets criteria).
    '''
    # choose points that are evenly spaced around each ring
    ex_pos = np.linspace(-np.pi, np.pi, n_ex)

    # choose real saddle or stable points
    stable_saddle_idx = (stable_idx | saddle_idx)
    stable_saddle_idx[im_idx] = False

    # index by map
    map_1_idx = np.round(fp_dist, 1) < -map_thresh
    map_2_idx = np.round(fp_dist, 1) > map_thresh
    middle_idx = (np.round(fp_dist, 1) > -1 + map_thresh) \
                    & (np.round(fp_dist, 1) < 1 - map_thresh)

    # set the example point for each position bin
    m1_idx = np.zeros(n_ex)
    m2_idx = np.zeros(n_ex)
    mid_idx = np.zeros(n_ex)
    for i, p in enumerate(ex_pos):
        p_idx = np.abs(pos_pred_fp - p) <= 0.2
        
        # map 1
        m1_options = np.where(p_idx & 
                              map_1_idx & 
                              stable_saddle_idx)[0]
        if m1_options.shape[0] == 0:
            m1_idx[i] = np.nan # no fps near this pos
        else:
            m1_idx[i] = np.random.choice(m1_options)
        
        # map 2
        m2_options = np.where(p_idx & 
                              map_2_idx & 
                              stable_saddle_idx)[0]
        if m2_options.shape[0] == 0:
            m2_idx[i] = np.nan # no fps near this pos
        else:
            m2_idx[i] = np.random.choice(m2_options)
        
        # saddle
        mid_options = np.where(p_idx & 
                               middle_idx & 
                               stable_saddle_idx)[0]
        if mid_options.shape[0] == 0:
            mid_idx[i] = np.nan # no fps near this pos
        else:
            mid_idx[i] = np.random.choice(mid_options)
        
    ex_idx = np.concatenate((m1_idx, mid_idx, m2_idx))
    ex_idx = ex_idx[~np.isnan(ex_idx)].astype(int)

    return ex_idx, m1_idx, mid_idx, m2_idx


def get_top_eigvecs(eig_vals, eig_vecs):
    '''
    Pulls out the eigenvector associated with the largest
    eigenvalue for each fixed point.
    '''
    num_fixed_pts = eig_vals.shape[0]
    for i in range(num_fixed_pts):
        lam = eig_vals[i]
        V = eig_vecs[i]
        max_idx = np.argmax(lam.real)
        if i == 0:
            Vs = V[:, max_idx]
        else:
            Vs = np.column_stack((Vs, V[:, max_idx]))
    return Vs


def get_im_idx(eig_vals, eig_vecs):
    '''
    Indices for fixed points whose largest eigenvalue
    is imaginary.
    '''
    im_idx = np.asarray([])
    for i in range(num_fixed_pts):
        lam = eig_vals[i]
        V = eig_vecs[i]
        max_idx = np.argmax(lam.real)
        if np.abs(lam[max_idx].imag) > 1e-5:
            im_idx = np.append(im_idx, i)
    return im_idx.astype(int)

''' Nonlinear systems analysis 
Sussillo and Barak (Neural Comput., 2013)
Maheswaranathan et al. (Adv. Neural Inf. Process. Syst., 2019)
'''
def characterize_fps(model, task_params, \
                    fixed_pts, sort_eigs=True):
    """
    Linearizes RNN dynamics around each fixed point and takes 
    the eigendecomposition to approximate the local dynamics.
    
    Params
    ------
    fixed_pts : torch array, shape (num_points, hidden_size)
        locations in activity space for each fixed point
    sort_eigs : bool, default is True
        if True, sorts the output from largest to smallest eigenvalue
    
    Returns
    -------
    Js : ndarray, shape (num_fixed_pts, num_units, num_units)
        Jacobian for each fixed point
    max_eigs : ndarray, shape (num_fixed_pts, )
        real component for the largest eigenvalue for each fixed point
    eig_vals : ndarray, len (num_fixed_pts, num_units)
        the eigenvalues for each fixed point
    eig_vecs : ndarray, len (num_fixed_pts, num_units, num_units)
        the eigenvectors for each fixed point
    """
    # data params
    num_maps = task_params["num_maps"]
    num_pos_dims = 1
    # num_pos_dims = task_params["num_spatial_dimensions"]
    num_fixed_pts = fixed_pts.shape[0]

    # get the Jacobian for each fixed point
    inp_vel = torch.zeros(num_pos_dims)
    inp_remaps = torch.zeros(num_maps) 
    Js = []
    for i in trange(num_fixed_pts):
        fp = fixed_pts[i]
        J = compute_jacobian(model, fp, \
                             inp_vel=inp_vel, \
                             inp_remaps=inp_remaps)
        Js.append(J.detach().numpy())
    Js = np.asarray(Js)

    # take the eigendecomposition of the Jacobians
    eig_vals = []
    eig_vecs = []
    max_eigs = np.asarray([])
    for i, J in enumerate(Js):
        lam, V = np.linalg.eig(J)
        eig_vals.append(lam)
        eig_vecs.append(V)
        max_eigs = np.append(max_eigs, np.max(lam.real))
    eig_vals = np.asarray(eig_vals)
    eig_vecs = np.asarray(eig_vecs)

    # sort by the max eigenvalue
    if sort_eigs:
        sort_idx = np.argsort(max_eigs).astype(int)
        sort_idx = sort_idx[::-1].astype(int)

        fixed_pts = fixed_pts[sort_idx]
        Js = Js[sort_idx]
        eig_vals = eig_vals[sort_idx]
        eig_vecs = eig_vecs[sort_idx]
        max_eigs = max_eigs[sort_idx]

    return Js, max_eigs, eig_vals, eig_vecs, sort_idx

def compute_jacobian(model, x, **kwargs):
    """
    Compute Jacobian of a fixed point, `x`, by passing it through
    one step of the model, with 0 velocity or context input.
    """
    # We need to track gradients / differentiate with
    # respect to x.
    x = x.detach()
    x = x.detach()    
    x.requires_grad_(True)

    # Define space for the Jacobian.
    x_1 = model.one_step(prev_state=x, **kwargs)
    num_outputs = x_1.size()[0]
    num_inputs = x.size()[0]
    J = torch.empty((num_outputs, num_inputs))

    for i in range(num_outputs):

        # Define the function we want to differentiate
        # locally at input x.
        y = model.one_step(prev_state=x, **kwargs)

        # Create a one-hot vector at position i.
        z = torch.zeros(num_inputs)
        z[i] = 1

        # Compute (z.T @ J) where J is the jacobian of
        # the function we care about at x.
        y.backward(z)

        # For this choice of z, the backward pass yields
        # the i-th row of the Jacobian matrix.
        J[i] = x.grad

        # Reset the gradient of x to zero.
        x.grad.zero_()

    return J


''' Geometric analysis '''
def dist_to_map(X1, X2, y):
    '''
    Determine where each element of y falls 
    along the context tuning dimension.

    Params
    ------
    X1, X2 : ndarray, shape (n_obs, hidden_size)
        neural firing in each map
    y : ndarray, shape (n_pts, hidden_size)
        points in activity space

    Returns
    -------
    y_dist : ndarray, shape (n_pts,)
        distance along the remapping dim to each map
        -1 = in map 1; 1 = in map 2; 0 = between the maps
    '''
    # mean center the activity in each map
    X1_bar = np.mean(X1, axis=0)
    X2_bar = np.mean(X2, axis=0)

    # define the normalized remapping dimension (hidden_size,)
    remap_dim = X1_bar - X2_bar / np.linalg.norm(X1_bar - X2_bar)

    # project onto the remapping dim
    proj_m1 = X1_bar @ remap_dim # map 1
    proj_m2 = X2_bar @ remap_dim # map 2
    proj_y = y @ remap_dim # (n_pts,)

    # get distance to map
    y_dist = (proj_y - proj_m1) / (proj_m2 - proj_m1)
    return 2 * (y_dist - .5) # classify -1 or 1

def align_eigvecs(eig_vals, eig_vecs, fp_dist, \
                    X, map_targ, pos_targ):
    '''
    Find the alignment of the top eigenvectors to the
    context tuning dimension and position subspaces.

    Each eigenvector is projected onto the position
    subspace associated with the closest map, or for
    saddle points onto the joint position subspace.

    For complex top eigenvectors, first define a plane
    spanned by the real and imaginary components and
    then align that plane to each dimension.

    Params
    ------
    eig_vals : ndarray, len (num_fixed_pts, num_units)
        the eigenvalues for each fixed point
    eig_vecs : ndarray, len (num_fixed_pts, num_units, num_units)
        the eigenvectors for each fixed point
    fp_dist : ndarray, shape (num_fixed_pts,)
        distance along the remapping dim to each map
        -1 = in map 1; 1 = in map 2; 0 = between the maps
    X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    pos_targ : ndarray, shape (n_obs,)
        track positions at each observation (rad)
    map_targ : ndarray of ints, shape (n_obs,)
        context at each observation

    Returns
    -------
    remap_angles : ndarray, shape (num_fixed_pts, )
        angle between each top eigenvector and the
        context tuning dimension
    pos_angles : ndarray, shape (num_fixed_pts, )
        angle between each top eigenvector and the
        nearest position subspace
    '''
    num_fixed_pts = eig_vals.shape[0]

    # find imaginary top eigenvals & store the top eigenvecs
    im_idx = get_im_idx(eig_vals, eig_vecs)
    Vs = get_top_eigvecs(eig_vals, eig_vecs)

    # split the activity and positions by context
    X0 = X[map_targ==0]
    X1 = X[map_targ==1]
    pos0 = pos_targ[map_targ==0]
    pos1 = pos_targ[map_targ==1]

    # define the context tuning dimension
    remap_dim = remapping_dim(X0, X1)

    # define the position subspaces
    X0_tc, _ = tuning_curve_1d(X0, pos0, \
                                n_pos_bins=250)
    X1_tc, _ = tuning_curve_1d(X1, pos1, \
                                n_pos_bins=250)
    m1_subspace = position_subspace(X0_tc)
    m2_subspace = position_subspace(X1_tc)
    pos_subspace = position_subspace(np.stack((X0_tc, X1_tc)))

    # index by nearest map
    map_1_idx = np.round(fp_dist, 1) < -0.5
    map_2_idx = np.round(fp_dist, 1) > 0.5

    # relate each eigenvector to each subspace
    remap_angles = np.asarray([])
    pos_angles = np.asarray([])
    for i in trange(num_fixed_pts):
        v = Vs[:, i]
        
        # compare to the appropriate pos subspace
        if map_1_idx[i]:
            p_subspace = m1_subspace
        elif map_2_idx[i]:
            p_subspace = m2_subspace
        else:
            p_subspace = pos_subspace
        
        # if complex, define the plane spanned by its eigenvectors
        if i in im_idx:
            u = v.real - np.mean(v.real)
            w = v.imag - np.mean(v.imag)
            u_norm = u /  np.linalg.norm(u)
            w_norm = w / np.linalg.norm(w)
            v_sub = np.stack((u_norm, w_norm), axis=-1)
            
            r_angle = dim_alignment.proj_aB(remap_dim, v_sub)
            p_angle = subspace_angles(p_subspace, v_sub)
            p_angle = np.cos(np.max(p_angle))
            
        else:
            u = v - np.mean(v)
            u_norm = u / np.linalg.norm(u)
            r_angle = dim_alignment.cosine_sim(remap_dim, u_norm)
            p_angle = dim_alignment.proj_aB(u_norm, p_subspace)
        
        remap_angles = np.append(remap_angles, r_angle)
        pos_angles = np.append(pos_angles, p_angle)

    return remap_angles, pos_angles