import sys
sys.path.append("../utils/")
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

from model_utils import sample_rnn_data, format_rnn_data
from basic_analysis import tuning_curve_1d, compute_misalignment
import analysis_rnn as rnn
from analysis_neuro import spatial_similarity

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.special import softmax
from scipy import stats

def plot_a(data_folder, model_IDs):
    '''
    Plot fold-change in firing rate versus spatial dissimilarity across maps
    for all 2-map models.

    Params:
    ------
    dissim : ndarray
        1 - cosine similarity across maps; shape (n_cells, )
    pct_FR : ndarray
        percent change in peak firing rate; shape (n_cells, )
    '''
    # data params
    n_models = len(model_IDs)
    dissim = np.asarray([])
    pct_dFR = np.asarray([])
    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                                targets["map_targets"],\
                                                targets["pos_targets"])
        n_units = X.shape[-1]


        # get the avg firing rates in each map
        fr_0, _ = tuning_curve_1d(X[map_targ==0],
                                    pos_targ[map_targ==0],
                                    smooth=True)
        fr_1, _ = tuning_curve_1d(X[map_targ==1],
                                    pos_targ[map_targ==1],
                                    smooth=True)
        pct_dFR = np.append(pct_dFR, rnn.pct_change_FR(fr_0, fr_1))
        dissim = np.append(dissim, rnn.spatial_dissimilarity(fr_0, fr_1))
    assert(dissim.shape[0] == n_models * n_units)
    assert(pct_dFR.shape[0] == n_models * n_units)

    # remove nans
    nan_idx = np.isnan(dissim) | np.isnan(pct_dFR)
    dissim = dissim[~nan_idx]
    pct_dFR = pct_dFR[~nan_idx]

    print(f'N models = {n_models}, N total units = {dissim.shape[0]}')

    # figure params
    gs = gridspec.GridSpec(6, 6, hspace=0, wspace=0)
    f = plt.figure(figsize=(3, 3)) 
    PT_SIZE = 5
    LW_THRESH = 1.5
    LW_HIST = 1.5

    # plot change in FR vs. dissimilarity, all cells
    ax0 = plt.subplot(gs[2:, :-2])
    ax0.scatter(dissim, pct_dFR, color='k', s=PT_SIZE, lw=0, alpha=0.2)
    ax0.set_xlim(0, 1.05)
    ymax = ax0.get_ylim()[1] + 25
    ax0.set_xlabel('spatial dissimilarity', fontsize=10, labelpad=1)
    ax0.set_ylabel('fold change in\npeak firing rate', fontsize=10, labelpad=1)

    # plot density, spatial
    ax1 = plt.subplot(gs[:2, :-2])
    n1, bins1, _ = ax1.hist(dissim, bins=50, density=True, histtype='stepfilled', 
                          lw=LW_HIST, edgecolor='k', facecolor='xkcd:light gray', alpha=0.7)
    ax1.set_xlim(0, 1.05)
    ymax_ax1 = ax1.get_ylim()[1]
    ax1.tick_params(labelbottom=False, which='major', labelsize=8, pad=0.5)
    ax1.set_ylabel('% cells', fontsize=10, labelpad=2)

    # plot density, firing rate
    ax2 = plt.subplot(gs[2:, -2:])
    n2, bins2, _ = ax2.hist(pct_dFR, bins=50, density=True, histtype='stepfilled', orientation='horizontal', 
                          lw=LW_HIST, edgecolor='k', facecolor='xkcd:light gray', alpha=0.7)
    ax2.tick_params(labelleft=False, which='major', labelsize=8, pad=0.5)
    ax2.set_xlabel('% cells', fontsize=10, labelpad=2)
    xmax = ax2.get_xlim()[1]

    # plot medians etc
    ax0.vlines(np.median(dissim), 0, ymax, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax1.vlines(np.median(dissim), 0, ymax_ax1, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax0.vlines(np.percentile(dissim, 95), 0, ymax, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax1.vlines(np.percentile(dissim, 95), 0, ymax_ax1, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax0.hlines(np.median(pct_dFR), 0, 1.05, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax2.hlines(np.median(pct_dFR), 0, xmax, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1, label='median')
    ax0.hlines(np.percentile(pct_dFR, 95), 0, 1.05, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax2.hlines(np.percentile(pct_dFR, 95), 0,  xmax, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1, label='95$^{th}$ pct')

    print(f'median change in firing rate = {(np.median(pct_dFR)/100)+1:.2}-fold')
    print(f'95th percentile change in firing rate = {(np.percentile(pct_dFR, 95)/100)+1:.2}-fold')
    print(f'median spatial dissimilarity = {np.median(dissim):.2}')
    print(f'95th percentile dissimilarity = {np.percentile(dissim, 95):.2}')

    # lims and labels
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    ax0.set_xticks([0, 0.5, 1])
    ax0.set_xticklabels([0, 0.5, 1])
    ax0.set_yticks([0, 100, 200])
    ax0.set_yticklabels([1, 2, 3])
    ax0.set_ylim(-2, ymax)
    ax2.set_ylim(-2, ymax)
    ax0.tick_params(which='major', labelsize=8, pad=0.5)

    # make the hist ticks meaningful
    b1_width = np.unique(np.round(np.diff(bins1), 8))
    vals = np.arange(0, 55, 25)
    t1s = np.zeros(vals.shape)
    for i, v in enumerate(vals):
        t1s[i] = v/(100*b1_width)
    ax1.set_yticks(t1s)
    ax1.set_yticklabels(vals)

    b2_width = np.unique(np.round(np.diff(bins2), 8))
    vals = np.arange(2, 7, 2)
    t2s = np.zeros(vals.shape)
    for i, v in enumerate(vals):
        t2s[i] = v/(100*b2_width)
    ax2.set_xticks(t2s)
    ax2.set_xticklabels(vals)
    # ax2.legend(bbox_to_anchor=(0.8,0.9,0,0), fontsize=7.5)

    return f, gs