import numpy as np
import torch
import sys
sys.path.append("../utils/")

from scipy import stats

from basic_analysis import tuning_curve_1d
from dim_alignment import position_subspace, remapping_dim, cosine_sim, proj_aB
from model_utils import load_model_params, sample_rnn_data, format_rnn_data


''' position and remapping dims alignment to the inputs and outputs '''
def align_remap_dims(data_folder, model_IDs):
    '''
    For each model, find the angle between the  remapping dimensions
    '''
    avg_angle = np.zeros(len(model_IDs))
    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"], \
                                                targets["map_targets"], \
                                                targets["pos_targets"])
        
        n_maps = np.max(np.unique(map_targ)) + 1
        remap_dims = []
        for j in range(n_maps):
            # pairwise comparison of maps
            m0_id = j
            m1_id = (j+1)%n_maps

            # activity by context
            X0 = X[map_targ==m0_id]
            X1 = X[map_targ==m1_id]

            # find the remapping dimension
            remap_dims.append(remapping_dim(X0, X1))
        
        angles = np.zeros(n_maps)
        for j in range(n_maps):
            angles[j] = cosine_sim(remap_dims[j], \
                                    remap_dims[(j+1)%n_maps])
        avg_angle[i] = np.rad2deg(np.arccos(np.abs(np.mean(angles))))
    
    model_avg = int(np.mean(avg_angle))
    model_sem = stats.sem(avg_angle)
    print(f'angle between remapping dims (mean, sem) = {model_avg} deg., {model_sem:.2} deg.')

def align_in_out(data_folder, model_IDs):
    '''
    For each model, determines the alignment between the
    context/position inputs and outputs and the position and remapping dimensions
    pairwise between each pair of maps
    '''
    inputs, outputs, targets = sample_rnn_data(data_folder, model_IDs[0])
    X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"], \
                                            targets["map_targets"], \
                                            targets["pos_targets"])
    n_maps = np.max(np.unique(map_targ)) + 1
    n_models = len(model_IDs)

    # to store the projections
    remap_dim_angles = {
        "ctxt_in": np.zeros((n_models, n_maps, 2)),
        "ctxt_out": np.zeros((n_models, n_maps, 2)),
        "pos_in": np.zeros((n_models, n_maps)), 
        "pos_out": np.zeros((n_models, n_maps, 2))
    }
    pos_dim_angles = {
        "ctxt_in": np.zeros((n_models, n_maps, 2)),
        "ctxt_out": np.zeros((n_models, n_maps, 2)),
        "pos_in": np.zeros((n_models, n_maps)), 
        "pos_out": np.zeros((n_models, n_maps, 2))
    }

    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        model, _, _ = load_model_params(data_folder, m_id)
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"], \
                                                targets["map_targets"], \
                                                targets["pos_targets"])
        n_maps = np.max(np.unique(map_targ)) + 1
        
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
        
        for j in range(n_maps):
            # pairwise comparison of maps
            m0_id = j
            m1_id = (j+1)%n_maps

            # activity by context
            X0 = X[map_targ==m0_id]
            X1 = X[map_targ==m1_id]

            # find the remapping dimension
            remap_dim = remapping_dim(X0, X1)

            # find the position subspace
            X0_tc, _ = tuning_curve_1d(X0, \
                                        pos_targ[map_targ==m0_id], \
                                        n_pos_bins=250)
            X1_tc, _ = tuning_curve_1d(X1, \
                                        pos_targ[map_targ==m1_id], \
                                        n_pos_bins=250)
            pos_subspace = position_subspace(np.stack((X0_tc, X1_tc)))

            # find the alignments
            ctxt_inp_w_maps = ctxt_inp_w[:, [m0_id, m1_id]]
            ctxt_out_w_maps = ctxt_inp_w[:, [m0_id, m1_id]]
            all_weights = [ctxt_inp_w_maps, \
                            ctxt_out_w_maps, \
                            pos_inp_w, \
                            pos_out_w]
            for label, w in zip(remap_dim_angles.keys(), all_weights):
                remap_dim_angles[label][i, j] = np.abs(cosine_sim(remap_dim, w))
                pos_dim_angles[label][i, j] = np.abs(proj_aB(w, pos_subspace))

    return remap_dim_angles, pos_dim_angles