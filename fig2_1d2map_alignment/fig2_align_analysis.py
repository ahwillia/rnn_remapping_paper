import numpy as np
import torch
import scipy

import sys
sys.path.append("../utils/")
from basic_analysis import tuning_curve_1d
from dim_alignment import position_subspace, remapping_dim, cosine_sim, proj_aB
from model_utils import load_model_params, sample_rnn_data, format_rnn_data

def shuff_xi_rotate(tc_0, tc_1, W):
    ''' shuffle function for remap vectors '''
    n_pos_bins, n_units = tc_1.shape
    
    # randomly rotate tc_1 in the nullspace of W
    P = scipy.linalg.null_space(W.T)
    Q = np.linalg.qr(np.random.randn(P.shape[1], P.shape[1]))[0]
    tc_1_rotate = tc_1 @ (P @ Q @ P.T)
    
    # find random vectors
    return tc_0 - tc_1_rotate


def shuff_xi_nullspace(xi_p, W):
    ''' alt shuffle function for remap vectors '''
    n_pos_bins, n_units = xi_p.shape
    
    # find random vectors
    z = np.random.randn(n_units, n_pos_bins)

    # get the portion in the nullspace
    proj = W @ np.linalg.inv(W.T @ W) @ W.T
    z_null = z - proj @ z

    # normalize to each pos bin
    z_null = (z_null / np.linalg.norm(z_null)) * np.linalg.norm(xi_p, axis=1)
    
    return z_null.T


def compute_remap_vectors(data_folder, m_id, \
                            n_pos_bins=50, \
                            n_shuffle=100):
    '''
    Finds the true remapping vectors between each pair of
    position bins for a given model.

    Also computes the ideal remap vector (if perfectly aligned)
    and shuffled vectors (rotate tc_1 in the nullspace of W).

    Returns
    -------
    xi_p : ndarray, shape (n_pos_bins, n_units)
        vector separating each pair of position bins
    v : ndarray, shape (n_pos_bins, n_units)
        vector separating the centers of the two ring manifolds
    z_shuff : ndarray, shape (n_shuffle, n_pos_bins, n_units)
        xi_p for random rotations of one ring
    '''
    model, _, _ = load_model_params(data_folder, m_id)
    inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
    X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                            targets["map_targets"],\
                                            targets["pos_targets"])
    n_units = X.shape[-1]

    # split by context
    X0 = X[map_targ==0]
    X1 = X[map_targ==1]
    pos0 = pos_targ[map_targ==0]
    pos1 = pos_targ[map_targ==1]

    # get the position-binned firing rates
    tc_0, _ = tuning_curve_1d(X0, pos0, n_pos_bins=n_pos_bins)
    tc_1, _ = tuning_curve_1d(X1, pos1, n_pos_bins=n_pos_bins)

    # find each xi_p
    xi_p = tc_0 - tc_1

    # find the ideal remap vector
    v = remapping_dim(tc_0, tc_1)

    # find the shuffle vectors
    pos_out_w = model.readout_layer_pos.weight
    W = pos_out_w.detach().numpy().T
    z_shuff = np.zeros([n_shuffle, n_pos_bins, n_units])
    for k in range(n_shuffle):
        # shuffle: randomly rotate tc_1
        z_shuff[k] = shuff_xi_rotate(tc_0, tc_1, W)

    return xi_p, v, z_shuff

def remap_vector_geometry(data_folder, m_id, **kwargs):
    '''
    Finds the difference between the true and ideal remap vectors,
    as well as the shuffled difference.

    Also determines the angle between the vectors and the 
    position subspace.

    Returns
    -------
    d_xi : ndarray, shape (n_pos_bins,)
        difference between the ideal and true remap vectors
        norm(v - xi_p) / norm(v) 
    avg_Wxi : ndarray, shape (n_pos_bins,)
        angle between xi_p and the position subspace
        avg((xi_p @ W) / norm(xi_p))
    d_shuff : ndarray, shape (n_shuffle, n_pos_bins)
        shuffle where xi_p is recomputed after randomly rotating
        tuning curve 1 in the nullspace of W.
    '''
    # get the rnn data
    model, _, _ = load_model_params(data_folder, m_id)

    # find the true and ideal remap vectors
    xi_p, v, z_shuff = compute_remap_vectors(data_folder, m_id, \
                                                **kwargs)

    # is xi_p orthog. to the position output weights?
    pos_out_w = model.readout_layer_pos.weight
    W = pos_out_w.detach().numpy().T
    Wxi = (xi_p @ W) / np.linalg.norm(xi_p)
    avg_Wxi = np.mean(Wxi, axis=1)

    # get the difference
    d_xi = np.linalg.norm(v - xi_p, axis=1) / np.linalg.norm(v)

    # compute the shuffles
    n_shuffle, n_pos_bins, n_units = z_shuff.shape
    d_shuff = np.zeros([n_shuffle, n_pos_bins])
    for k in range(n_shuffle):
        # compare the shuffles to the ideal
        z = z_shuff[k]
        d_shuff[k] = np.linalg.norm(v - z, axis=1) / np.linalg.norm(v)
        
    return avg_Wxi, d_xi, d_shuff