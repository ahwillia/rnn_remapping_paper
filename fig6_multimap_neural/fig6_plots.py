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

from analysis_neuro import tuning_curve
from fig6_analysis import load_neural_data, format_neural_data

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.special import softmax
from scipy import stats

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
    Example of remapping for a 3-map session:
    raster and TC for one unit
    network-wide similarity and distance to cluster

    Params:
    ------
    d : dict
        data for the example mouse/session
    cell_ID : int
        ID number for the example cell.
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

    # set indices for each map
    map0_idx = d['idx'][0, :]
    map1_idx = d['idx'][1, :]

    # figure parameters
    gs = gridspec.GridSpec(8, 7, hspace=1.2, wspace=4)
    f = plt.figure(figsize=(1.6, 1)) 
    PT_SIZE = 0.3   
    LW_MEAN = 0.5
    LW_SEM = 0.1   
    CLU_W = 4 

    # plot raster and tuning curves colored by map
    ax2 = plt.subplot(gs[:-2, :3]) # raster
    ax3 = plt.subplot(gs[-2:, :3]) # tuning curves
    for j in range(n_maps):
        # raster
        map_idx = d['idx'][j, :].copy()
        sdx = B[map_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
        ax2.scatter(A[map_idx, 0][sdx], A[map_idx, 2][sdx], \
                    color=all_map_colors[j], lw=0, s=PT_SIZE, alpha=.1)
        
        # tuning curves with SEM
        sdx = (np.where(cells==cell_ID)[0][0]).astype(int)
        FR_map = np.squeeze(all_FR[j])
        sem_map = np.squeeze(all_FR_sem[j])
        ax3.plot(FR_map[:, sdx], color=all_map_colors[j], lw=LW_MEAN, alpha=1)
        ax3.fill_between(binned_pos/2, FR_map[:, sdx] + sem_map[:, sdx], \
                            FR_map[:, sdx] - sem_map[:, sdx], \
                            color=all_map_colors[j], linewidth=LW_SEM, alpha=0.2)

    # axes and lims
    trial_max = np.round(np.max(A[:, 2]), -1)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_bounds(0, 20)
    ax3.spines['bottom'].set_bounds(0, 200)
    ax2.set_xlim((0, 400))
    ylim_ax = [0, np.max(A[:, 2])]
    ax2.set_ylim(ylim_ax[::-1])
    ax3.set_ylim([0, 25])
    ax3.set_xlim([0, 200])

    ax2.set_yticks([0, trial_max//2, trial_max])
    ax2.set_xticks([0, 200, 400])
    ax3.set_xticks([0, 100, 200])
    ax3.set_xticklabels([0, 200, 400])
    ax3.set_yticks([0, 20])
    ax2.tick_params(labelbottom=False, which='major',\
                    labelsize=tick_label, pad=0.5)
    ax3.tick_params(which='major', labelsize=tick_label, pad=0.5)
    
    # labels
    ax2.set_title('ex. cell', fontsize=title_size, pad=3)    
    ax2.set_ylabel('trial', fontsize=axis_label, labelpad=1)
    ax3.set_ylabel('FR', fontsize=axis_label, labelpad=6)
    ax3.set_xlabel('pos. (cm)', fontsize=axis_label, labelpad=1)

    # plot similarity matrix
    ax1 = plt.subplot(gs[:-2, 3:])
    im = ax1.imshow(sim, clim=[0, 1], aspect='auto', cmap='gist_yarg')
    ax1.set_title("network", fontsize=title_size, pad=3)
    ax1.tick_params(labelleft=False, which='major', 
                    labelsize=tick_label, pad=0.5)
    ax1.set_yticks([0, trial_max//2, trial_max])
    ax1.set_xticks([0, trial_max//2, trial_max])
    ax1.set_xlabel("map", fontsize=axis_label, labelpad=5)

    # plot cluster assignments
    ax0 = plt.subplot(gs[-1, 3:])
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


def plot_b(firing_rates, binned_pos,
    num_points=1000,
    axlim=2,
    reflect_x=False,
    reflect_y=False,
    reflect_z=False):
    
    '''
    Ring manifolds for the neural data.

    Params
    ------
    firing_rates : ndarray, shape (n_trials, n_pos_bins, n_cells)
        firing rate by position bin and trial for all neurons
    binned_pos : ndarray, shape (n_pos_bins,)
        centers of each position bin (cm)
    '''

    # data params
    n_trials, n_pos_bins, n_cells = firing_rates.shape

    # get the k-means cluster centroids
    kmeans = KMeans(n_clusters=2, n_init=50, random_state=1234)
    M = firing_rates.reshape(n_trials, -1)
    kmeans.fit(M)
    m1, m2 = kmeans.cluster_centers_.reshape(2, n_pos_bins, n_cells)

    # mean center
    m1m2 = np.row_stack((m1.copy(), m2.copy()))
    m1m2_bar = m1m2 - np.mean(m1m2, axis=0)

    # Find PCs
    pca = PCA(n_components=3)
    x_, y_, z_ = pca.fit_transform(m1m2_bar).T

    # Reflect axes if desired
    if reflect_x:
        x_ *= -1
    if reflect_y:
        y_ *= -1
    if reflect_z:
        z_ *= -1

    # fig params
    fig = plt.figure(figsize=(3, 2.1))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')
    DOT_SIZE = 30
    PC_LW = 3
    SHADOW_LW = 6

    # plot activity
    ax.scatter(
        x_[:n_pos_bins], y_[:n_pos_bins], z_[:n_pos_bins],
        c=binned_pos, cmap=ring_colormap(),
        alpha=1, lw=0, s=DOT_SIZE)
    ax.scatter(
        x_[n_pos_bins:], y_[n_pos_bins:], z_[n_pos_bins:],
        c=binned_pos, cmap=ring_colormap(),
        alpha=1, lw=0, s=DOT_SIZE)

    # plot shadow
    ax.plot(
        x_[:n_pos_bins], y_[:n_pos_bins], 
        np.full(n_pos_bins, -axlim),
        color="k", alpha=.1, lw=SHADOW_LW)
    ax.plot(
        x_[n_pos_bins:], y_[n_pos_bins:], 
        np.full(n_pos_bins, -axlim),
        color="k", alpha=.1, lw=SHADOW_LW)

    # axis params
    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)
    ax.set_zlim(-axlim, axlim)

    # plot axes
    axlim = axlim - 1
    pc1 = np.asarray([[-axlim, axlim], [axlim, axlim], [-axlim, -axlim]])
    pc2 = np.asarray([[axlim, axlim], [-axlim, axlim], [-axlim, -axlim]])
    pc3 = np.asarray([[axlim, axlim], [axlim, axlim], [-axlim, axlim]])
    for p in [pc1, pc2, pc3]:
        p[0] = p[0] + 1
        p[1] = p[1] + 2

    ax.plot(*pc1, color="k", alpha=.8, lw=PC_LW)
    ax.plot(*pc2, color="k", alpha=.8, lw=PC_LW)
    ax.plot(*pc3, color="k", alpha=.8, lw=PC_LW)

    ax.set_title('neural manifolds',
                  fontsize=title_size, pad=-10)
    ax.view_init(azim=135, elev=40)
    ax.axis("off")

    return fig, ax


def plot_c(data_folder, session_IDs, num_maps):
    '''
    Misalignment scores for manifolds from all multimap sessions
    Scores are normalized:
        0 = perfectly aligned
        1 = 2.5% of shuffle (i.e., p = 0.025)

    num_maps : ndarray, shape (n_sessions,)
        number of maps in each session
    '''
    # data params
    n_sessions = len(session_IDs)
    n_maps = np.max(num_maps)
    dt = 0.02 # time bin
    pos_bin = 2 # cm
    n_pos_bins = 400 // pos_bin

    alignment_scores = np.zeros((n_sessions, n_maps))
    alignment_scores.fill(np.nan)
    for i, s_id in enumerate(session_IDs):
        n_maps = num_maps[i]

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
            FRs[j], _, _ = tuning_curve(A[m_idx, 0],
                                        B[m_idx, :],
                                        dt, b=2, SEM=True)

        # get the manifolds for each map and compute misalignment
        for j in range(n_maps):
            m0_id = j
            m1_id = (j+1)%n_maps
            norm_align, _, _, _ = compute_misalignment(FRs[m0_id], FRs[m1_id])
            alignment_scores[i, j] = norm_align

    # define 1:2 as most aligned, 3:1 as least aligned
    sort_alignment = np.sort(alignment_scores, axis=1)
    flat_alignment = alignment_scores.ravel()
    flat_alignment = flat_alignment[~np.isnan(flat_alignment)]

    print(f'mean misalignment = {np.mean(flat_alignment):.2}')
    print(f'sem misalignment = {stats.sem(flat_alignment):.2}')

    # fig params 
    f, ax = plt.subplots(1, 1, figsize=(1, 1))
    DOT_LW = 0.5
    DOT_SIZE = 5
    THRESH_LW = 2
    JIT = np.random.randn(n_sessions) * 0.03 # jitter points
    n_maps = np.max(num_maps)

    # plot the alignment scores for each pair
    # V = ax.violinplot(sort_alignment, 
    #                   showextrema=False)
    # for i, v in enumerate(V['bodies']):
    #     v.set_facecolor(all_map_colors[i])
    for j in range(n_maps):
        ax.scatter(np.full(n_sessions, j+1)+JIT, 
                   sort_alignment[:, j], 
                   facecolors='w', edgecolors=all_map_colors[j], 
                   s=DOT_SIZE, lw=DOT_LW, alpha=1)
        
    # plot the shuffle threshold
    ax.plot([0.8, 4.1], [1, 1], dashes=[1, 1], lw=THRESH_LW, color="k")
    ax.text(5, 1, "shuff\nthresh", fontsize=tick_label,\
            horizontalalignment='center', verticalalignment='center')
        
    # ticks and lims
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 1)
    ax.spines['bottom'].set_bounds(1, 4)
    ax.set_xlim([0.6, 4.5])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim([0, 1.2])
    ax.set_yticks(np.arange(0, 1.4, 0.5))
    ax.tick_params(which='major', labelsize=tick_label, pad=0.5)

    # labels
    ax.set_ylabel('misalignment', fontsize=axis_label, labelpad=1)
    ax.set_xticklabels(['1:2', '2:3', '3:1', '4:1'])

    return f, ax