import sys
sys.path.append("../utils/")
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from plot_utils import simple_cmap, ring_colormap

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
c2 = 'k'

# rnn position input/output colors
pos_col = 'xkcd:cobalt blue'
est_col = 'xkcd:saffron'

def plot_fig1c(d, cell_ID, FR_0, FR_1, FR_0_sem, FR_1_sem, binned_pos):
    '''
    From Low et al, 2021
    Example of remapping for a single session:
    raster and TC for one unit
    network-wide similarity and distance to cluster

    Params:
    ------
    d : dict
        data for the example mouse/session
    cell_ID : int
        ID number for the example cell.
    FR_0, FR_1 : ndarray, shape (n_pos_bins, n_cells)
        firing rate by position within each map    
    FR_0_sem, FR_1_sem : ndarray, shape (n_pos_bins, n_cells)
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

    # set indices for each map
    map0_idx = d['idx'][0, :]
    map1_idx = d['idx'][1, :]

    # figure parameters
    gs = gridspec.GridSpec(8, 7, hspace=1.2, wspace=4)
    f = plt.figure(figsize=(1.6, 1))    
    LW_MEAN = 0.5
    LW_SEM = 0.1   
    CLU_W = 4 

    # plot raster
    ax2 = plt.subplot(gs[:-2, :3])
    sdx_0 = B[map0_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
    ax2.scatter(A[map0_idx, 0][sdx_0], A[map0_idx, 2][sdx_0],\
                color=c1, lw=0, s=0.3, alpha=.3)
    sdx_1 = B[map1_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
    ax2.scatter(A[map1_idx, 0][sdx_1], A[map1_idx, 2][sdx_1],\
                color=c2, lw=0, s=0.3, alpha=.2)
    ax2.set_xlim((0, 400))
    ylim_ax = [0, np.max(A[:, 2])]
    ax2.set_ylim(ylim_ax[::-1])
    ax2.set_yticks([0, 200, 400])
    ax2.set_ylabel('trial', fontsize=axis_label, labelpad=1)
    ax2.tick_params(labelbottom=False, which='major',\
                    labelsize=tick_label, pad=0.5)
    ax2.set_title('ex. cell', fontsize=title_size, pad=3)

    # plot tuning curves with SEM
    sdx = (np.where(cells==cell_ID)[0][0]).astype(int)
    ax3 = plt.subplot(gs[-2:, :3])
    ax3.plot(FR_0[:, sdx], c1, lw=LW_MEAN, alpha=0.9)
    ax3.fill_between(binned_pos/2,\
                        FR_0[:, sdx] + FR_0_sem[:, sdx],\
                        FR_0[:, sdx] - FR_0_sem[:, sdx],\
                        color=c1, linewidth=LW_SEM, alpha=0.3)
    ax3.plot(FR_1[:, sdx], color=c2, lw=LW_MEAN, alpha=1)
    ax3.fill_between(binned_pos/2,\
                        FR_1[:, sdx] + FR_1_sem[:, sdx],\
                        FR_1[:, sdx] - FR_1_sem[:, sdx],\
                        color=c2, linewidth=LW_SEM, alpha=0.4)
    
    ax2.set_xticks([0, 200, 400])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_bounds(0, 12)
    ax3.spines['bottom'].set_bounds(0, 200)
    ax3.set_xticks([0, 100, 200])
    ax3.set_xticklabels([0, 200, 400])
    ax3.set_yticks([0, 12])
    ax3.set_ylim([0, 15])
    ax3.set_xlim([0, 200])
    ax3.set_ylabel('FR', fontsize=axis_label, labelpad=6)
    ax3.set_xlabel('pos. (cm)', fontsize=axis_label, labelpad=1)
    ax3.tick_params(which='major', labelsize=tick_label, pad=0.5)

    # plot similarity matrix
    ax1 = plt.subplot(gs[:-2, 3:])
    im = ax1.imshow(sim, clim=[0.1, 0.7], aspect='auto', cmap='Greys')
    ax1.set_title("network", fontsize=title_size, pad=3)
    ax1.tick_params(labelleft=False, which='major', 
                    labelsize=tick_label, pad=0.5)
    ax1.set_yticks([0, 200, 400])
    ax1.set_xticks([0, 200, 400])
    ax1.set_xlabel("map", fontsize=axis_label, labelpad=5)

    # plot cluster assignments
    ax0 = plt.subplot(gs[-1, 3:])

    all_map_colors = ['xkcd:red', c2]
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


def plot_fig1d(firing_rates, binned_pos,
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


def plot_fig1e(inp_vel, inp_remaps):
    '''
    Inputs for the 1D navigation/working memory task.

    Params
    ------
    inp_vel : ndarray, shape (num_steps, 1)
        input velocity for an example trial
    inp_remaps : ndarray, shape (num_steps, num_maps)
        binary context cues for an example trial
    '''

    # fig params
    gs = gridspec.GridSpec(4, 1, hspace=0)
    f = plt.figure(figsize=(1.8, 1.8))
    VEL_LW = 1
    CUE_LW = 1

    # plot velocity input
    ax0 = plt.subplot(gs[0])
    ax0.plot(inp_vel,
                c=pos_col, lw=VEL_LW)
    ax0.set_title('velocity input',
                      fontsize=title_size, pad=5)
    ax0.set_axis_off()

    # white space
    ax1 = plt.subplot(gs[1])
    ax1.set_axis_off()

    # plot context cue inputs
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax2.set_title('binary context cues',
                      fontsize=title_size, pad=0)
    ax2.plot(inp_remaps[:, 0], 
             c=c1, lw=CUE_LW)
    ax3.plot(inp_remaps[:, 1],
             c=c2, lw=CUE_LW)
    ax2.set_ylim([-0.2, 1.5])
    ax3.set_ylim([-0.2, 1.5])
    ax2.set_axis_off()
    ax3.set_axis_off()

    return f, gs


def plot_fig1f(targ, pred, map_logits):
    '''
    Outputs for the 1D navigation/working memory task.

    Params
    ------
    targ : ndarray, shape (num_steps,)
        true positions for an example trial
    pred : ndarray, shape (num_steps,)
        estimated positions for an example trial
    map_logits : ndarray, shape (num_steps, num_maps)
        context estimates for an example trial
    '''

    # fig params
    gs = gridspec.GridSpec(4, 1, hspace=0)
    f = plt.figure(figsize=(1.8, 1.8))
    TRUE_POS_LW = 2
    EST_POS_LW = 1.5
    CTXT_LW = 1

    # plot pos output vs. true pos
    ax0 = plt.subplot(gs[0])
    ax0.plot(targ, c=pos_col,
             lw=TRUE_POS_LW, 
             zorder=0)
    ax0.plot(pred, c=est_col,
             dashes=[1, 1], lw=EST_POS_LW,
             zorder=1)
    ax0.set_title('position output',
                      fontsize=title_size, pad=4)
    xlims = ax0.get_xlim()
    ax0.hlines([np.pi, -np.pi], xlims[0], xlims[1], 
               colors='k', alpha=0.6,
               linestyles='dashed', lw=EST_POS_LW)
    ax0.set_axis_off()

    # white space
    ax1 = plt.subplot(gs[1])
    ax1.set_axis_off()

    # plot context cue inputs
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax2.set_title('context estimate',
                      fontsize=title_size, pad=0)
    ax2.plot(softmax(map_logits, axis=1)[:, 0], 
             c=c1, lw=CTXT_LW)
    ax3.plot(softmax(map_logits, axis=1)[:, 1],
             c=c2, lw=CTXT_LW)
    ax2.set_ylim([-0.2, 1.5])
    ax3.set_ylim([-0.2, 1.5])
    ax2.set_axis_off()
    ax3.set_axis_off()

    return f, gs


def plot_fig1g(pos_losses, map_losses):
    '''
    Final loss for position and context estimates across models
    (see task.py for the loss functions)

    Params
    ------
    pos_losses : ndarray, shape (n_models,)
        final model loss for position estimate vs. true position
    map_losses : ndarray, shape (n_models,)
        final model loss for context estimate vs. true context
    '''
    n_models = pos_losses.shape[0]

    # figure params
    f, ax = plt.subplots(1, 1, figsize=(0.5, 1))
    pos_col = 'xkcd:cobalt blue'
    c1 = 'xkcd:scarlet'
    DOT_SIZE = 10
    DOT_LW = 1

    # for jittering points
    JIT = np.random.randn(n_models) * 0.03

    # 1 = pos loss, 2 = map loss
    ax.scatter(np.full(n_models, 1)+JIT, 
               pos_losses,
               c=pos_col,
               s=DOT_SIZE, lw=DOT_LW,
               alpha=0.5
              )
    ax.scatter(np.full(n_models, 2)+JIT, 
               map_losses,
               c=c1,
               s=DOT_SIZE, lw=DOT_LW,
               alpha=0.5
              )

    # axis params
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 0.06)
    ax.spines['bottom'].set_bounds(1, 2)

    ax.set_xlim([0.75, 2.5])
    ax.set_ylim([-0.002, 0.05])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['pos\nest', 'map\nest'])
    ax.set_yticks([0, 0.03, 0.06])
    ax.set_ylabel('final loss', fontsize=axis_label, labelpad=1)

    ax.tick_params(which='major', labelsize=tick_label, pad=0.5)

    return f, ax


def plot_fig1h(firing_rates, network_similarity, map_idx, ex_units):
    '''
    Replicate fig. 1b for 6 example model units

    Params
    ------
    firing_rates : ndarray, shape (n_traversals, n_units, n_pos_bins)
        position-binned RNN unit activity for each track traversal
    network_similarity : ndarray, shape (n_traversals, n_traversals)
        trial-trial correlation of spatial activity across units
    map_idx : ndarray, shape (n_traversals, n_maps)
        which map was predominant on each traversal
        1, predominant map; 0, other map(s)
    '''
    # data params
    n_ex_units = ex_units.shape[0]
    (n_traversals, n_units, n_pos_bins) = firing_rates.shape
    binned_pos = np.linspace(0, 2*np.pi, num=n_pos_bins)

    # define remaps
    remap_idx = np.asarray([])
    for m in range(map_idx.shape[1]):
        remaps = np.where(np.abs(np.diff(map_idx[:, m])))[0]
        remap_idx = np.append(remap_idx, remaps)
    remap_idx = np.unique(remap_idx)
    remap_idx = remap_idx.astype(int)
    start_idx = np.append([0], remap_idx)
    end_idx = np.append(remap_idx, n_traversals)

    # get the colors for map IDs
    all_map_colors = [c1, c2]
    map_colors = []
    for i in np.where(map_idx[remap_idx, :])[1]:
        map_colors.append(all_map_colors[i])
    map_colors.append(all_map_colors[np.where(map_idx[-1, :])[0][0]])

    # figure params
    width_ratios = np.append(np.full(n_ex_units, 3), 4)
    f, ax = plt.subplots(2, n_ex_units + 1,
                         figsize=(0.85*n_ex_units, 1),
                         gridspec_kw=dict(height_ratios=[3.5, 1],
                                          width_ratios=width_ratios,
                                          hspace=0.15, wspace=0.2))   
    DOT_SIZE = 0.3
    LW_MEAN = 0.5
    LW_SEM = 0.1   
    CLU_W = 3 
    fr_cmap = simple_cmap(c2, 'w', c1)

    for i, u in enumerate(ex_units):
        # set axes
        ax0 = ax[0, i]
        ax1 = ax[1, i]
        
        # plot firing rates by traversal/map
        fr = firing_rates[:, u]
        fr_1 = fr.copy()
        fr_1[map_idx[:, 1].astype(bool)] = 0
        fr_2 = fr.copy()
        fr_2[map_idx[:, 0].astype(bool)] = 0
        fr_split = fr_1 - fr_2
        ax0.imshow(fr_split, aspect='auto', cmap=fr_cmap, clim=[-0.8, 0.8])

        # plot tuning curves with SEM
        avg_fr_1 = np.mean(fr_1, axis=0)
        sem_fr_1 = stats.sem(fr_1, axis=0)
        ax1.plot(binned_pos, avg_fr_1, c1, lw=LW_MEAN, alpha=1)
        ax1.fill_between(binned_pos, avg_fr_1 + sem_fr_1, avg_fr_1 - sem_fr_1,
                         color=c1, linewidth=LW_SEM, alpha=0.4)
        avg_fr_2 = np.mean(fr_2, axis=0)
        sem_fr_2 = stats.sem(fr_2, axis=0)
        ax1.plot(binned_pos, avg_fr_2, c2, lw=LW_MEAN, alpha=1)
        ax1.fill_between(binned_pos, avg_fr_2 + sem_fr_2, avg_fr_2 - sem_fr_2,
                         color=c2, linewidth=LW_SEM, alpha=0.4)

        # ticks and lims
        ax0.set_xticks([0, n_pos_bins/2, n_pos_bins])
        ax0.set_yticks([0, np.round(n_traversals/2, -2), np.round(n_traversals, -2)])
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_bounds(0, 0.4)
        ax1.spines['bottom'].set_bounds(0, 2*np.pi)
        ax1.set_xlim([0, 2*np.pi])
        ax1.set_xticks([0, np.pi, 2*np.pi])
        ax1.set_ylim([0, 0.5])
        ax1.set_yticks([0, 0.4])
        
        # labels
        ax1.set_xticklabels([0, '$\pi$', '2$\pi$'])
        ax0.tick_params(labelbottom=False, which='major', labelsize=tick_label, pad=0.5)
        ax1.tick_params(which='major', labelsize=tick_label, pad=0.5)
        if i == 0:
            ax0.set_ylabel('trial', fontsize=axis_label, labelpad=1)
            ax1.set_ylabel('FR', fontsize=axis_label, labelpad=0.5)
        else:
            ax0.tick_params(labelleft=False)
            ax1.tick_params(labelleft=False)
        if i == n_ex_units//2:
            ax0.set_title('example RNN units', fontsize=title_size, pad=3)
            ax1.set_xlabel('track position (rad.)', fontsize=axis_label, labelpad=1)

    # plot similarity matrix
    ax2 = ax[0, -1]
    im = ax2.imshow(network_similarity, \
                    clim=[0.1, 0.7], \
                    aspect='auto', \
                    cmap='Greys')
    ax2.set_title("network", fontsize=title_size, pad=3)
    ax2.tick_params(labelleft=False, which='major', 
                    labelsize=tick_label, pad=0.5)
    ax2.set_yticks([0, np.round(n_traversals/2, -2), np.round(n_traversals, -2)])
    ax2.set_xticks([0, np.round(n_traversals/2, -2), np.round(n_traversals, -2)])
    ax2.set_xlabel("map", fontsize=axis_label, labelpad=5)

    # plot map labels (by trial)
    ax3 = ax[1, -1]
    ax3.hlines(np.full(start_idx.shape[0], 0.7), start_idx, end_idx, 
                colors=map_colors, lw=np.full(start_idx.shape[0], CLU_W), 
                linestyles=np.full(start_idx.shape[0], 'solid'))
    ax3.set_ylim([0.5, 1.5])
    ax3.set_xlim([0, n_traversals])
    plt.axis('off')    
    ax3.tick_params(which='major', labelsize=tick_label, pad=0.5)

    return f, ax


def plot_fig1i(X, pos_targets,
    num_points=1000,
    axlim=2,
    reflect_x=False,
    reflect_y=False,
    reflect_z=False):
    
    '''
    Ring manifolds for the 1D navigation/working memory task.

    Params
    ------
    X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    pos_targets : ndarray, shape(n_obs,)
        true positions at each observation
    '''

    # data params
    num_posbins, num_neurons = X.shape
    idx = np.random.choice(num_posbins, size=num_points, replace=False)

    # mean center
    X_bar = X - np.mean(X, axis=0)

    # Find PCs
    pca = PCA(n_components=3)
    x_, y_, z_ = pca.fit_transform(X_bar).T

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
    DOT_SIZE = 10
    PC_LW = 3

    # plot activity
    ax.scatter(
        x_[idx], y_[idx], z_[idx],
        c=pos_targets[idx], cmap=ring_colormap(),
        alpha=0.6, lw=0, s=DOT_SIZE)

    # plot shadow
    ax.scatter(
        x_[idx], y_[idx], 
        np.full(num_points, -axlim),
        color="k", alpha=.02, lw=0, s=DOT_SIZE)

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

    ax.set_title('RNN activity manifolds',
                  fontsize=title_size, pad=-10)
    ax.view_init(azim=130, elev=30)
    ax.axis("off")

    return fig, ax


# TODO plot_fig1j - ring alignment histogram


def plot_fig1k(X, map_targets, \
                top_num=4, top_num_1=3, \
                most_var_thresh=0.90):
    '''
    Variance explained by each component for both maps together
    and for each map alone (insets).

    Params
    ------
    X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    map_targets : ndarray, shape(n_obs,)
        true context at each observation
    top_num : int
        number of PCs to highlight for both maps
    top_num_1 : int
        number of PCs to highlight for each map
    most_var_thresh : float
        threshold for how many components explain "most of the variance"
    '''
    # data params
    hidden_size = X.shape[-1]

    # PCA on both manifolds
    pca = PCA().fit(X)
    var = pca.explained_variance_
    total_var = np.sum(var)
    pct_var = (var / total_var)

    # PCA on each manifold
    X0 = X[map_targets==0]
    X1 = X[map_targets==1]

    pca_0 = PCA().fit(X0)
    var_0 = pca_0.explained_variance_
    pct_var_0 = (var_0 / np.sum(var_0))

    pca_1 = PCA().fit(X1)
    var_1 = pca_1.explained_variance_
    pct_var_1 = (var_1 / np.sum(var_1))

    # print the results
    cum_var = np.cumsum(pct_var)
    most_var = np.argmin(np.abs(cum_var - most_var_thresh))
    print(f'{3} PCs explain {cum_var[2]:.2%} of the variance')
    print(f'{most_var+1} PCs explain {cum_var[most_var]:.2%} of the variance')

    # figure params
    f = plt.figure(figsize=(1.5, 0.8))
    DOT_SIZE = 5
    DOT_LW = 0.2
    CUM_LW = 1

    # both manifolds
    ax0 = plt.axes([0, 0, 1, 1])
    ax0.plot(np.arange(hidden_size) + 1,
             np.cumsum(pct_var),
             c='k', lw=CUM_LW,
             zorder=0)
    ax0.scatter(np.arange(hidden_size) + 1,
                np.cumsum(pct_var),
                c='k', s=DOT_SIZE-2, lw=0,
                zorder=1)
    ax0.scatter(np.arange(top_num) + 1,
                np.cumsum(pct_var)[:top_num],
                facecolors='r', edgecolors='k',
                s=DOT_SIZE, lw=DOT_LW,
                zorder=2, label=f'{top_num} PCs')

    # ticks and labels
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['left'].set_bounds(0, 1)
    ax0.spines['bottom'].set_bounds(1, hidden_size)
    ax0.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax0.set_xlim((-5, hidden_size+5))
    ax0.set_xticks([1, hidden_size/2, hidden_size])
    ax0.set_ylim((-0.05, 1.05))
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_title('both maps',
                  fontsize=title_size, pad=4)
    ax0.set_xlabel('dimension', fontsize=axis_label, labelpad=1)
    ax0.set_ylabel('variance\nexplained', 
                   fontsize=axis_label, labelpad=1)

    # map 1
    ax1 = plt.axes([0.2, 0.25, 0.35, 0.4])
    ax1.plot(np.arange(hidden_size) + 1,
             np.cumsum(pct_var_0),
             c='k', lw=CUM_LW/2,
             zorder=0)
    ax1.scatter(np.arange(hidden_size) + 1,
                np.cumsum(pct_var_0),
                c='k', s=DOT_SIZE-3, lw=0,
                zorder=1)
    ax1.scatter(np.arange(top_num_1) + 1,
                np.cumsum(pct_var_0)[:top_num_1],
                facecolors='r', edgecolors='k',
                s=DOT_SIZE-2, lw=DOT_LW,
                zorder=2)

    # ticks and labels
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_bounds(0, 1)
    ax1.spines['bottom'].set_bounds(1, hidden_size)
    ax1.tick_params(which='major', labelsize=tick_label-1, pad=0.5)
    ax1.set_xlim((-7, hidden_size+5))
    ax1.set_ylim((-0.05, 1.05))
    ax1.set_xticks([1, hidden_size])
    ax1.set_yticks([0, 1])
    ax1.set_title('map 1',
                  fontsize=axis_label, pad=3)

    # map 2
    ax2 = plt.axes([0.63, 0.25, 0.35, 0.4])
    ax2.plot(np.arange(hidden_size) + 1,
             np.cumsum(pct_var_1),
             c='k', lw=CUM_LW/2,
             zorder=0)
    ax2.scatter(np.arange(hidden_size) + 1,
                np.cumsum(pct_var_1),
                c='k', s=DOT_SIZE-3, lw=0,
                zorder=1)
    ax2.scatter(np.arange(top_num_1) + 1,
                np.cumsum(pct_var_1)[:top_num_1],
                facecolors='r', edgecolors='k',
                s=DOT_SIZE-2, lw=DOT_LW,
                zorder=2, label=f'{top_num_1} PCs')

    # ticks and labels
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_bounds(0, 1)
    ax2.spines['bottom'].set_bounds(1, hidden_size)
    ax2.tick_params(labelleft=False, which='major', labelsize=tick_label-1, pad=0.5)
    ax2.set_xlim((-7, hidden_size+5))
    ax2.set_ylim((-0.05, 1.05))
    ax2.set_xticks([1, hidden_size])
    ax2.set_yticks([0, 1])
    ax2.set_title('map 2',
                  fontsize=axis_label, pad=3)

    return f, [ax0, ax1, ax2]