import numpy as np
import torch
import sys
sys.path.append("../utils/")

from basic_analysis import tuning_curve_2d
from dim_alignment import position_subspace, remapping_dim, cosine_sim, proj_aB
from model_utils import load_model_params, sample_rnn_data, format_rnn_data

from scipy import stats

''' position and remapping dims alignment to the inputs and outputs '''
def align_in_out(data_folder, model_IDs):
    n_models = len(model_IDs)

    # to store the projections
    remap_dim_angles = {
        "ctxt_in": np.zeros((n_models, 2)),
        "ctxt_out": np.zeros((n_models, 2)),
        "pos_in": np.zeros(n_models, 2), 
        "pos_out": np.zeros((n_models, 4))
    }
    pos_dim_angles = {
        "ctxt_in": np.zeros((n_models, 2)),
        "ctxt_out": np.zeros((n_models, 2)),
        "pos_in": np.zeros(n_models, 2), 
        "pos_out": np.zeros((n_models, 4))
    }

    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        model, _, _ = load_model_params(data_folder, m_id)
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"], \
                                                targets["map_targets"], \
                                                targets["pos_targets"])
        
        # split by context
        X0 = X[map_targ==0]
        X1 = X[map_targ==1]
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
        ctxt_inp_w = model.linear_ih.weight[:, 1:]
        pos_inp_w = model.linear_ih.weight[:, 0]
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