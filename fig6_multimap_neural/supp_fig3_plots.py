import sys
sys.path.append("../utils/")
sys.path.append("../fig1_1d2map/")
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from plot_utils import simple_cmap, ring_colormap
from model_utils import sample_rnn_data, format_rnn_data
from basic_analysis import tuning_curve_1d, compute_misalignment
from fig6_plots import plot_waveforms

from analysis_neuro import tuning_curve
from fig6_analysis import get_cell_channels, wf_correlations

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.special import softmax
from scipy import stats
import itertools

''' general figure params '''
# font sizes
title_size = 10
axis_label = 9
tick_label = 7

# map colors
c1 = 'xkcd:scarlet'
c2 = 'xkcd:green blue'
c3 = 'k'
c4 = 'xkcd:saffron'
all_map_colors = [c2, c1, c3, c4]

def plot_a(d, cell_ID, all_FR, all_FR_sem, binned_pos):
    '''
    Example session showing remapping:
    raster and TC for one unit
    network-wide similarity and distance to cluster

    Params:
    ------
    d : dict
        data for the example mouse/session
    cell_ID : array of ints
        ID numbers for the example cells.
    all_FR : ndarray, shape (n_maps, n_pos_bins, n_cells)
        firing rate by position within each map    
    all_FR_sem : ndarray, shape (n_maps, n_pos_bins, n_cells)
        SEM for the firing rate arrays
    binned_pos : ndarray, shape (n_pos_bins,)
        centers of each position bin (cm)
    '''

    # load relevant data
    A = d['A'] # behavior
    B = d['B'] # spikes
    cells = d['cells'] # cell IDs
    sim = d['sim'] # trial-trial similarity
    remap_idx = d['remap_idx'] # remap trial index
    W = d['kmeans']['W']

    # data params
    n_maps = W.shape[1]
    n_ex_cells = cell_ID.shape[0]

    # set indices for each map
    map0_idx = d['idx'][0, :]
    map1_idx = d['idx'][1, :]

    # figure parameters
    gs = gridspec.GridSpec(8, 14, hspace=1.2, wspace=4)
    f = plt.figure(figsize=(3.8, 1.2)) 
    PT_SIZE = 0.4   
    LW_MEAN = 0.5
    LW_SEM = 0.1   
    CLU_W = 4 

    # plot raster and tuning curves colored by map
    col_start = 5
    for i, c in enumerate(cell_ID):
        col_end = col_start + 3
        ax2 = plt.subplot(gs[:-2, col_start:col_end]) # raster
        ax3 = plt.subplot(gs[-2:, col_start:col_end]) # tcs
        for j in range(n_maps):
            # raster
            map_idx = d['idx'][j, :].copy()
            sdx = B[map_idx, np.where(cells==c)[0][0]].astype(bool)
            ax2.scatter(A[map_idx, 0][sdx], A[map_idx, 2][sdx], \
                        color=all_map_colors[j], lw=0, s=PT_SIZE, alpha=.1)
            
            # tuning curves with SEM
            sdx = (np.where(cells==c)[0][0]).astype(int)
            FR_map = np.squeeze(all_FR[j])
            sem_map = np.squeeze(all_FR_sem[j])
            ax3.plot(FR_map[:, sdx], color=all_map_colors[j], lw=LW_MEAN, alpha=1)
            ax3.fill_between(binned_pos/2, FR_map[:, sdx] + sem_map[:, sdx], \
                                FR_map[:, sdx] - sem_map[:, sdx], \
                                color=all_map_colors[j], linewidth=LW_SEM, alpha=0.2)
        # axes and lims
        trial_max = np.round(np.max(A[:, 2]), -1)
        if i==0:
            ax2.tick_params(labelbottom=False,\
                            which='major', labelsize=tick_label, pad=0.5)
            ax3.tick_params(which='major', labelsize=tick_label, pad=0.5)
            ax3.set_ylabel('FR', fontsize=axis_label, labelpad=0.2)
        else:
            ax2.tick_params(labelbottom=False, labelleft=False,\
                            which='major', labelsize=tick_label, pad=0.5)
            ax3.tick_params( labelleft=False,\
                            which='major', labelsize=tick_label, pad=0.5)
        if i==1:
            ax3.set_xlabel('position (cm)', fontsize=axis_label, labelpad=1)

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['left'].set_bounds(0, 25)
        ax3.spines['bottom'].set_bounds(0, 200)
        ax2.set_xlim((0, 400))
        ylim_ax = [0, np.max(A[:, 2])]
        ax2.set_ylim(ylim_ax[::-1])
        ax3.set_ylim([0, 35])
        ax3.set_xlim([0, 200])

        ax2.set_yticks([0, trial_max//2, trial_max])
        ax2.set_xticks([0, 200, 400])
        ax3.set_xticks([0, 100, 200])
        ax3.set_xticklabels([0, 200, 400])
        ax3.set_yticks([0, 25])
        ax2.set_title(f'example\ncell {i+1}', fontsize=title_size, pad=3)    

        col_start = col_end

    # plot similarity matrix
    ax1 = plt.subplot(gs[:-2, :4])
    im = ax1.imshow(sim, clim=[0, 1], aspect='auto', cmap='gist_yarg')
    ax1.set_title("network\nsimilarity", fontsize=title_size, pad=3)
    ax1.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax1.set_yticks([0, trial_max//2, trial_max])
    ax1.set_xticks([0, trial_max//2, trial_max])
    ax1.set_ylabel('trial', fontsize=axis_label, labelpad=1)
    ax1.set_xlabel("map", fontsize=axis_label, labelpad=6)

    # plot cluster assignments
    ax0 = plt.subplot(gs[-1, :4])
    start_idx = np.append([0], remap_idx)
    end_idx = np.append(remap_idx, W.shape[0])
    map_colors = []
    for i in np.where(W[remap_idx, :])[1]:
        map_colors.append(all_map_colors[i])
    map_colors.append(all_map_colors[np.where(W[-1, :])[0][0]])

    ax0.hlines(np.full(start_idx.shape[0], 1), start_idx, end_idx, 
                colors=map_colors, lw=np.full(start_idx.shape[0], CLU_W), 
                linestyles=np.full(start_idx.shape[0], 'solid'))
    ax0.set_ylim([0.5, 1.5])
    ax0.set_xlim([0, W.shape[0]])
    plt.axis('off')    
    ax0.tick_params(which='major', labelsize=tick_label, pad=0.5)

    return f, gs

def plot_b(d, cell_ID, session_idx=1):
    # data params
    d['cell_channels'] = get_cell_channels(d)
    n_maps, n_cells, _, _ = d['waveform_avg'].shape

    # get the waveform correlations, median, and percetiles
    avg_corr = wf_correlations(d)
    pct_5 = np.percentile(avg_corr, 5)
    pct_95 = np.percentile(avg_corr, 95)
    med_corr = np.median(avg_corr)

    # figure parameters
    gs = gridspec.GridSpec(8, 14, hspace=1.2, wspace=4)
    f = plt.figure(figsize=(3.8, 1.2)) 
    PT_SIZE = 0.4   
    LW_MEAN = 0.5
    LW_SEM = 0.1   
    CLU_W = 4 
    DOT_LW = 0.5
    DOT_SIZE = 5
    BAR_SIZE = 4
    BAR_WIDTH = 0.8
    LW_PCT = 0.6
    JIT = np.random.randn(n_cells) * 0.05 # jitter points
    
    # plot the average waveform correlation across epochs for all cells
    ax = plt.subplot(gs[:, 1:4])
    ax.scatter(np.full(n_cells, 1)+JIT, avg_corr,\
               facecolors='none',\
               edgecolors=all_map_colors[1],\
               s=DOT_SIZE, lw=DOT_LW, alpha=1, zorder=1)
        
    # plot the median and 5/95th percentiles
    ax.plot(1, med_corr, '_k',\
            markersize=BAR_SIZE,\
            markeredgewidth=BAR_WIDTH,\
            zorder=2, alpha=1)
    ax.vlines(1, pct_5, pct_95, lw=LW_PCT,\
                colors='k', linestyles='solid',\
                zorder=2, alpha=1)
        
    # ticks and lims
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0.4, 1)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim([-0.1, 2.1])
    ax.set_ylim([0.3, 1.2])
    ax.set_xticks([])
    ax.set_yticks(np.arange(0.4, 1.1, 0.2))
    ax.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax.set_title('correlations\nall cells', fontsize=title_size, pad=3)

    # plot waveform overlay colored by map for each cell
    col_start = 5
    for i, c in enumerate(cell_ID):
        col_end = col_start + 3
        ax2 = plt.subplot(gs[:, col_start:col_end])
        for j in range(n_maps):
            plot_waveforms(d, c, j, ax2, \
                            all_map_colors[j])
        # plot black on top
        plot_waveforms(d, c, 2, ax2, \
                            all_map_colors[2])

        if i==1:
            ax2.set_title(f'waveform overlays\ncell {i+1}', fontsize=title_size, pad=3)
        else:
            ax2.set_title(f'cell {i+1}', fontsize=title_size, pad=3)
        col_start = col_end

    return f, gs