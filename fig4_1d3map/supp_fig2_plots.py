import sys
sys.path.append("../utils/")
import numpy as np

import matplotlib.pyplot as plt

from model_utils import sample_rnn_data, format_rnn_data
from basic_analysis import tuning_curve_1d, compute_misalignment
import fig4_analysis as rnn

from scipy import stats
import itertools

''' general figure params '''
# font sizes
title_size = 10
axis_label = 9
tick_label = 7

# colors
pos_col = 'xkcd:cobalt blue'
c1 = 'xkcd:scarlet'

def plot_c(data_folder, model_IDs, num_maps):
    '''
    average final loss for navigation and latent state estimation
    for different numbers of maps

    num_maps : ndarray, shape (n_models,)
        number of maps for each model
    '''
    # data params
    n_models = len(model_IDs)
    pos_losses = np.asarray([])
    map_losses = np.asarray([])
    for m_id in model_IDs:
        # get the final losses
        pos_loss = np.load(f"{data_folder}/{m_id}/pos_losses.npy")
        pos_losses = np.append(pos_losses, pos_loss[-1])
        map_loss = np.load(f"{data_folder}/{m_id}/map_losses.npy")
        map_losses = np.append(map_losses, map_loss[-1])

    # figure params
    f, ax = plt.subplots(2, 1, figsize=(2, 2))
    DOT_SIZE = 10

    # for jittering points
    JIT = np.random.randn(n_models) * 0.03

    # pos loss across different num maps
    ax[0].scatter(num_maps + JIT, 
                   pos_losses,
                   c=pos_col,
                   s=DOT_SIZE, lw=0,
                   alpha=0.5
                  )

    # map loss across different num maps
    ax[1].scatter(num_maps + JIT, 
                   map_losses,
                   c=c1,
                   s=DOT_SIZE, lw=0,
                   alpha=0.5
                  )

    for i in range(2):
        # ticks and lims
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_bounds(0, 0.06)
        ax[i].spines['bottom'].set_bounds(np.min(num_maps), np.max(num_maps))
        ax[i].set_xlim([np.min(num_maps)-0.3, np.max(num_maps)+1])
        ax[i].set_xticks([np.min(num_maps), 
                          (np.max(num_maps) + np.min(num_maps))//2, 
                          np.max(num_maps)])
        ax[i].set_ylim([-0.003, 0.05])
        ax[i].set_yticks([0, 0.03, 0.06])
        ax[i].tick_params(which='major', labelsize=tick_label, pad=0.5)
        
    # labels
    ax[0].set_ylabel('pos. loss', fontsize=axis_label, labelpad=1)
    ax[1].set_ylabel('map loss', fontsize=axis_label, labelpad=1)
    ax[0].tick_params(labelbottom=False)
    ax[1].set_xlabel('num. maps', fontsize=axis_label, labelpad=1)

    return f, ax

def plot_d(data_folder, model_IDs, num_maps):
    '''
    angle between adjacent pairs of remapping dimensions
    for different numbers of maps

    num_maps : ndarray, shape (n_models,)
        number of maps for each model
    '''
    # data params
    multi_map_models = []
    for i, m_id in enumerate(model_IDs):
        if num_maps[i] >= 3:
            multi_map_models.append(m_id)
    n_models = len(multi_map_models)
    M = num_maps[num_maps >= 3]

    all_angles = rnn.align_remap_dims(data_folder, multi_map_models, save_angles=True)

    # fig params
    f, ax = plt.subplots(1, 1, figsize=(2, 1))
    DOT_SIZE = 5
    ANGLE_LW = 1.5
    JIT = np.random.randn(n_models) * 0.03 # jitter points

    # plot the angles b/w remap dims by number of maps
    for i in range(n_models):
        angles = all_angles[i]
        n_angles = angles.shape[0]
        JIT = np.random.randn(n_angles) * 0.03 # jitter points
        ax.scatter(np.full(n_angles, M[i])+JIT, 
                   angles, 
                   facecolors=c1, 
                   s=DOT_SIZE, lw=0,
                   alpha=0.2, zorder=0)

    # plot the expected angles
    ax.plot([2.5, 10.5], [60, 60], dashes=[1, 1], 
                lw=ANGLE_LW, color="k")

    # ticks and lims
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 90)
    ax.spines['bottom'].set_bounds(M[0], M[-1])
    ax.set_xlim([M[0] - 0.3, M[-1] + 1])
    ax.set_xticks(np.arange(M[0], M[-1] + 1))
    ax.set_ylim([-1, 91])
    ax.set_yticks(np.arange(0, 100, 30))

    # labels
    ax.set_ylabel('angle between\n remap dims', fontsize=axis_label, labelpad=1)
    ax.set_xlabel('num. maps', fontsize=axis_label, labelpad=1)
    ax.tick_params(which='major', labelsize=tick_label, pad=0.5)

    return f, ax

def plot_e(data_folder, model_IDs, num_maps):
    '''
    pairwise misalignment scores for different numbers of maps

    num_maps : ndarray, shape (n_models,)
        number of maps for each model
    '''
    # data params
    n_models = len(model_IDs)
    n_maps = np.max(num_maps)
    n_pairs = (np.math.factorial(n_maps)) // \
                    (np.math.factorial(2)*np.math.factorial(n_maps-2))
    n_pos_bins = 50

    alignment_scores = np.zeros((n_models, n_pairs))
    alignment_scores.fill(np.nan)
    for i, m_id in enumerate(model_IDs):
        # define the map pairs
        n_maps = num_maps[i]
        map_ids = np.arange(n_maps)
        m_pairs = list(itertools.combinations(map_ids,2))

        # get the rnn data
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                                targets["map_targets"],\
                                                targets["pos_targets"])
        n_units = X.shape[-1]
        
        # get the tuning curves for each map
        FRs = np.zeros([n_maps, n_pos_bins, n_units])
        for j in range(n_maps):
            FRs[j], _ = tuning_curve_1d(X[map_targ==j], pos_targ[map_targ==j])

        # get the manifolds for each map and compute misalignment
        for j, (m0_id, m1_id) in enumerate(m_pairs):
            norm_align, _, _, _ = compute_misalignment(FRs[m0_id], FRs[m1_id])
            alignment_scores[i, j] = norm_align

    # summarize alignment overall
    flat_alignment = alignment_scores.ravel()
    flat_alignment = flat_alignment[~np.isnan(flat_alignment)]
    print(f'mean misalignment = {np.mean(flat_alignment):.2}')
    print(f'sem misalignment = {stats.sem(flat_alignment):.2}')

    # organize alignment scores by number of maps
    n_maps = np.max(num_maps)
    scores_by_maps = []
    for m in range(2, n_maps+1):
        scores = alignment_scores[num_maps==m]
        flat_scores = scores.ravel()
        scores_by_maps.append(flat_scores[~np.isnan(flat_scores)])

    # fig params 
    f, ax = plt.subplots(1, 1, figsize=(2, 1))
    DOT_LW = 0.5
    DOT_SIZE = 5
    THRESH_LW = 2
    JIT = np.random.randn(n_pairs) * 0.03 # jitter points

    # plot the misalignment scores by number of maps
    for i, n_maps in enumerate(np.unique(num_maps)):
        scores = scores_by_maps[i]
        n_scores = scores.shape[0]
        JIT = np.random.randn(n_scores) * 0.03 # jitter points
        ax.scatter(np.full(n_scores, n_maps)+JIT, 
                   scores, 
                   facecolors=c1, 
                   s=DOT_SIZE, lw=0,
                   alpha=0.2, zorder=0)
        
    # plot the shuffle threshold
    ax.plot([1.5, 10], [1, 1], dashes=[1, 1], lw=THRESH_LW, color="k")
    ax.text(11, 1, "shuff\nthresh", fontsize=tick_label,\
            horizontalalignment='center', verticalalignment='center')

    # ticks and lims
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 1)
    ax.spines['bottom'].set_bounds(num_maps[0], num_maps[-1])
    ax.set_xlim([num_maps[0] - 0.3, num_maps[-1] + 1])
    ax.set_xticks(np.arange(num_maps[0], num_maps[-1] + 1))
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.4, 0.5))
    ax.tick_params(which='major', labelsize=tick_label, pad=0.5)

    # labels
    ax.set_ylabel('misalignment', fontsize=axis_label, labelpad=1)
    ax.set_xlabel('num. maps', fontsize=axis_label, labelpad=1)

    return f, ax