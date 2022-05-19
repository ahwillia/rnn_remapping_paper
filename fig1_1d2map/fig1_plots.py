import sys
sys.path.append("../utils/")
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from plot_utils import simple_cmap, ring_colormap
from model_utils import sample_rnn_data, format_rnn_data
from basic_analysis import tuning_curve_1d, compute_misalignment
import analysis_rnn as rnn
from analysis_neuro import spatial_similarity

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

def plot_c(d, cell_ID, FR_0, FR_1, FR_0_sem, FR_1_sem, binned_pos):
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
                color=c2, lw=0, s=0.3, alpha=.3)
    sdx_1 = B[map1_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
    ax2.scatter(A[map1_idx, 0][sdx_1], A[map1_idx, 2][sdx_1],\
                color=c1, lw=0, s=0.3, alpha=.2)
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
    ax3.plot(FR_0[:, sdx], c2, lw=LW_MEAN, alpha=0.9)
    ax3.fill_between(binned_pos/2,\
                        FR_0[:, sdx] + FR_0_sem[:, sdx],\
                        FR_0[:, sdx] - FR_0_sem[:, sdx],\
                        color=c2, linewidth=LW_SEM, alpha=0.3)
    ax3.plot(FR_1[:, sdx], color=c1, lw=LW_MEAN, alpha=1)
    ax3.fill_between(binned_pos/2,\
                        FR_1[:, sdx] + FR_1_sem[:, sdx],\
                        FR_1[:, sdx] - FR_1_sem[:, sdx],\
                        color=c1, linewidth=LW_SEM, alpha=0.4)
    
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
    im = ax1.imshow(sim, clim=[0, 1], aspect='auto', cmap='viridis')
    ax1.set_title("network", fontsize=title_size, pad=3)
    ax1.tick_params(labelleft=False, which='major', 
                    labelsize=tick_label, pad=0.5)
    ax1.set_yticks([0, 200, 400])
    ax1.set_xticks([0, 200, 400])
    ax1.set_xlabel("map", fontsize=axis_label, labelpad=5)

    # plot cluster assignments
    ax0 = plt.subplot(gs[-1, 3:])

    all_map_colors = [c2, c1]
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


def plot_d(firing_rates, binned_pos,
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


def plot_e(inp_vel, inp_remaps):
    '''
    Inputs for the 1D navigation/working memory task.

    Params
    ------
    inp_vel : ndarray, shape (num_steps, 1)
        input velocity for an example trial
    inp_remaps : ndarray, shape (num_steps, num_maps)
        binary context cues for an example trial

    inp_vel : torch.tensor, shape (num_steps, batch_size)
        velocity inputs from generate_batch()
    inp_remaps : torch.tensor, shape (num_steps, batch_size, num_maps)
        binary context cues from generate_batch()
    '''
    # format inputs and extract a single trial
    inp_vel = inp_vel.detach().numpy()[:, 0]
    inp_remaps = inp_remaps.detach().numpy()[:, 0, :]

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


def plot_f(pos_targets, pos_outputs, map_logits):
    '''
    Outputs for the 1D navigation/working memory task.

    Params
    ------
    pos_targets : torch.tensor, shape (num_steps, batch_size)
        true positions from generate_batch()
    pos_outputs : torch.tensor, shape (num_steps, batch_size, 2)
        model estimated positions
    map_logits : torch.tensor, shape (num_steps, batch_size, num_maps)
        model estimated contexts
    '''
    # format data and extract a single example trial
    map_logits = map_logits.detach().numpy()[:, 0, :]
    targ = pos_targets.detach().numpy()[:, 0, :]
    targ = (targ + np.pi) % (2 * np.pi) - np.pi
    pred = pos_outputs.detach().numpy()[:, 0, :]
    pred = np.arctan2(pred[:, 1], pred[:, 0])

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


def plot_g(data_folder, model_ID):
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
    # generate a "mouse-like" session
    X, pos_targ, map_targ = rnn.get_mouselike_data(data_folder, model_ID)

    # label each observation by traversal number
    traversals_by_obs = rnn.traversal_num(pos_targ)
    print(f'{np.max(traversals_by_obs)+1} total track traversals')

    # find the predominant context for each traversal
    map_idx = rnn.map_by_traversal(map_targ, traversals_by_obs)
    n_remaps = np.sum(np.abs(np.diff(map_targ)))
    print(f'{n_remaps} total remapping events')

    # get the position-binned firing rates for each traversal
    firing_rates = rnn.fr_by_traversal(X, pos_targ, \
                                        traversals_by_obs, n_pos_bins=50)

    # calculate the trial-trial spatial similarity
    network_similarity = spatial_similarity(firing_rates.copy())

    # choose example units
    n_units = X.shape[-1]
    n_ex_units = 5
    possible_units = np.arange(n_units)
    possible_units = \
            possible_units[np.max(np.mean(firing_rates, axis=0), axis=1) > 0.3]
    ex_units = np.random.choice(possible_units, n_ex_units, replace=False)

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
                    cmap='viridis')
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


def plot_h(X, pos_targets,
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


def plot_j(data_folder, model_IDs, \
            top_num=4, top_num_1=3, \
            most_var_thresh=0.90):
    '''
    Average variance explained by each component across maps
    for both maps together and for each map alone (insets).

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
    n_models = len(model_IDs)
    inputs, outputs, targets = sample_rnn_data(data_folder, model_IDs[0])
    X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                            targets["map_targets"],\
                                            targets["pos_targets"])
    hidden_size = X.shape[-1]

    # figure params
    f = plt.figure(figsize=(0.7, 0.4))
    DOT_SIZE = 5
    DOT_LW = 0.2
    CUM_LW = 1

    all_var = np.zeros((n_models, hidden_size))
    all_var_0 = np.zeros((n_models, hidden_size))
    all_var_1 = np.zeros((n_models, hidden_size))
    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                                targets["map_targets"],\
                                                targets["pos_targets"])
        hidden_size = X.shape[-1]
        
        # PCA on both manifolds
        pca = PCA().fit(X)
        var = pca.explained_variance_
        total_var = np.sum(var)
        pct_var = (var / total_var)
        cum_var = np.cumsum(pct_var)
        all_var[i, :] = cum_var
        
        # PCA on each manifold
        X0 = X[map_targ==0]
        X1 = X[map_targ==1]

        pca_0 = PCA().fit(X0)
        var_0 = pca_0.explained_variance_
        pct_var_0 = (var_0 / np.sum(var_0))
        cum_var_0 = np.cumsum(pct_var_0)
        all_var_0[i, :] = cum_var_0

        pca_1 = PCA().fit(X1)
        var_1 = pca_1.explained_variance_
        pct_var_1 = (var_1 / np.sum(var_1))
        cum_var_1 = np.cumsum(pct_var_1)
        all_var_1[i, :] = cum_var_1

    # print the results
    avg_var = np.mean(all_var, axis=0)
    most_var = np.argmin(np.abs(avg_var - most_var_thresh))
    print(f'on average, {top_num} PCs explain {avg_var[top_num-1]:.2%} of the variance')
    print(f'on average, {most_var+1} PCs explain {avg_var[most_var]:.2%} of the variance')
        
    # plot the average across models
    avg_var = np.mean(all_var, axis=0)
    ax0 = plt.axes([0, 0, 1, 1])
    ax0.scatter(np.arange(top_num, hidden_size) + 1,
                avg_var[top_num:],
                c='k', s=DOT_SIZE-2,
                lw=0, zorder=1)
    ax0.scatter(np.arange(top_num) + 1,
                avg_var[:top_num],
                facecolors='r', edgecolors='k',
                s=DOT_SIZE, lw=DOT_LW, zorder=1)
    ax0.plot(np.arange(hidden_size) + 1, avg_var,
             c='k', lw=CUM_LW, zorder=0)
        
    # ticks and labels
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['left'].set_bounds(0, 1)
    ax0.spines['bottom'].set_bounds(-5, hidden_size)
    ax0.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax0.set_xlim((-5, hidden_size+5))
    ax0.set_xticks([1, hidden_size/2, hidden_size])
    ax0.set_ylim((0, 1.05))
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_yticklabels([0, 0.5, 1])
    # ax0.set_title('both maps',
    #               fontsize=title_size, pad=4)
    ax0.set_xlabel('dimension', fontsize=axis_label, labelpad=1)
    ax0.set_ylabel('var. exp.', 
                   fontsize=axis_label, labelpad=1)

    # # map 1
    # avg_var_0 = np.mean(all_var_0, axis=0)
    # ax1 = plt.axes([0.2, 0.25, 0.35, 0.4])
    # ax1.plot(np.arange(hidden_size) + 1,
    #          avg_var_0,
    #          c='k', lw=CUM_LW/1.5,
    #          zorder=0)
    # ax1.scatter(np.arange(top_num_1, hidden_size) + 1,
    #             avg_var_0[top_num_1:],
    #             c='k', s=DOT_SIZE-3, lw=0,
    #             zorder=1)
    # ax1.scatter(np.arange(top_num_1) + 1,
    #             avg_var_0[:top_num_1],
    #             facecolors='r', edgecolors='k',
    #             s=DOT_SIZE-2, lw=DOT_LW,
    #             zorder=2)

    # # ticks and labels
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['left'].set_bounds(0, 1)
    # ax1.spines['bottom'].set_bounds(1, hidden_size)
    # ax1.tick_params(which='major', labelsize=tick_label-1, pad=0.5)
    # ax1.set_xlim((-7, hidden_size+5))
    # ax1.set_ylim((-0.05, 1.05))
    # ax1.set_xticks([1, hidden_size])
    # ax1.set_yticks([0, 1])
    # ax1.set_title('map 1',
    #               fontsize=axis_label, pad=3)

    # # map 2
    # avg_var_1 = np.mean(all_var_1, axis=0)
    # ax2 = plt.axes([0.63, 0.25, 0.35, 0.4])
    # ax2.plot(np.arange(hidden_size) + 1,
    #          avg_var_1,
    #          c='k', lw=CUM_LW/1.5,
    #          zorder=0)
    # ax2.scatter(np.arange(top_num_1, hidden_size) + 1,
    #             avg_var_1[top_num_1:],
    #             c='k', s=DOT_SIZE-3, lw=0,
    #             zorder=1)
    # ax2.scatter(np.arange(top_num_1) + 1,
    #             avg_var_1[:top_num_1],
    #             facecolors='r', edgecolors='k',
    #             s=DOT_SIZE-2, lw=DOT_LW,
    #             zorder=2, label=f'{top_num_1} PCs')

    # # ticks and labels
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['left'].set_bounds(0, 1)
    # ax2.spines['bottom'].set_bounds(1, hidden_size)
    # ax2.tick_params(labelleft=False, which='major', labelsize=tick_label-1, pad=0.5)
    # ax2.set_xlim((-7, hidden_size+5))
    # ax2.set_ylim((-0.05, 1.05))
    # ax2.set_xticks([1, hidden_size])
    # ax2.set_yticks([0, 1])
    # ax2.set_title('map 2',
    #               fontsize=axis_label, pad=3)

    return f, ax0 # [ax0, ax1, ax2]


def plot_k(data_folder, model_IDs):
    '''
    Alignment of the input and output weights to the
    remapping dimension and position subspace
    '''
    # project the input and output weights onto each dimension
    remap_dim_angles, pos_dim_angles = rnn.align_in_out(data_folder, model_IDs)

    # get the means and standard deviation
    remap_dim_means = np.asarray([])
    remap_dim_sems = np.asarray([])
    pos_dim_means = np.asarray([])
    pos_dim_sems = np.asarray([])
    for label in remap_dim_angles.keys():
        remap_dim_means = np.append(remap_dim_means, \
                                    np.mean(remap_dim_angles[label]))
        remap_dim_sems = np.append(remap_dim_sems, \
                                    stats.tstd(remap_dim_angles[label].ravel()))
        pos_dim_means = np.append(pos_dim_means, \
                                    np.mean(pos_dim_angles[label]))
        pos_dim_sems = np.append(pos_dim_sems, \
                                    stats.tstd(pos_dim_angles[label].ravel()))

    # fig params
    f, ax = plt.subplots(1, 2, figsize=(2, 1))
    bar_colors = ['xkcd:dark gray', c1, pos_col, est_col]
    ERR_LW = 1.5
    xcoords = [1, 2, 4, 5]
        
    # plot projection onto remap dim
    ylims = ax[0].get_ylim()
    err_dict = {'ecolor': 'k', 'elinewidth': ERR_LW}
    ax[0].bar(xcoords, remap_dim_means,
              width=0.8, bottom=ylims[0],
              color=bar_colors, alpha=1, edgecolor='k',
              yerr=remap_dim_sems, error_kw=err_dict)

    # plot projection onto position subspace
    err_dict = {'ecolor': 'k', 'elinewidth': ERR_LW}
    ax[1].bar(xcoords, pos_dim_means,
              width=0.8, bottom=ylims[0],
              color=bar_colors, alpha=1, edgecolor='k',
              yerr=pos_dim_sems, error_kw=err_dict) 

    # ticks and lims
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_yticks(np.arange(0, 1.2, 0.5))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_bounds(0, 1)
        ax[i].spines['bottom'].set_bounds(xcoords[0] - 0.5,
                                          xcoords[-1] + 0.5)
        ax[i].set_xlim([xcoords[0] - 0.7,
                        xcoords[-1] + 0.7])
        ax[i].set_ylim([0, 1])

    # labels
    ax[0].tick_params(labelbottom=False, which='major',
                      labelsize=tick_label, pad=0.5)
    ax[1].tick_params(labelbottom=False, labelleft=False,
                      which='major', labelsize=tick_label, pad=0.5)
    ax[0].set_ylabel('cosine sim.', fontsize=axis_label, labelpad=1)
    ax[0].set_title('remap dim.', fontsize=axis_label, pad=3)
    ax[1].set_title('pos. dim.', fontsize=axis_label, pad=3)

    return f, ax


''' POSSIBLE SUPPLEMENTAL FIGS '''
def plot_supp_1(data_folder, model_IDs):
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
    # data params
    n_models = len(model_IDs)
    pos_losses = np.asarray([])
    map_losses = np.asarray([])
    for m_id in model_IDs:
        pos_loss = np.load(f"{data_folder}/{m_id}/pos_losses.npy")
        pos_losses = np.append(pos_losses, pos_loss[-1])
        map_loss = np.load(f"{data_folder}/{m_id}/map_losses.npy")
        map_losses = np.append(map_losses, map_loss[-1])
    print(f'mean +/- standard error of the mean:')
    print(f'position loss: {np.mean(pos_losses):.3} +/- {stats.sem(pos_losses):.3}')
    print(f'context loss: {np.mean(map_losses):.3} +/- {stats.sem(map_losses):.3}')

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


def plot_supp_2(data_folder, model_IDs):
    '''
    Misalignment scores for manifolds from all trained models
    Scores are normalized:
        1 = perfectly aligned
        0 = 2.5% of shuffle (i.e., p = 0.025)
    '''
    # data params
    n_models = len(model_IDs)
    alignment_scores = np.asarray([])
    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                                targets["map_targets"],\
                                                targets["pos_targets"])
        
        # get the manifolds for each map and compute misalignment
        X0_binned, binned_pos = tuning_curve_1d(X[map_targ==0],\
                                                pos_targ[map_targ==0])
        X1_binned, _ = tuning_curve_1d(X[map_targ==1],\
                                        pos_targ[map_targ==1])
        norm_align, _, _, _ = compute_misalignment(X0_binned, X1_binned)
        
        alignment_scores = np.append(alignment_scores, norm_align)

    print(f'mean alignment = {np.mean(alignment_scores):.2}')
    print(f'sem misalignment = {stats.sem(alignment_scores):.2}')

    # fig params 
    f, ax = plt.subplots(1, 1, figsize=(1, 0.5))
    BAR_LW = 1
    THRESH_LW = 2

    ax.hist(
        alignment_scores,
        np.linspace(0.0, 1.0, 30),
        color="gray", lw=BAR_LW, edgecolor="k")
    ax.set_xlabel("misalignment", fontsize=axis_label, labelpad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([1, 1], [0, 3.3], dashes=[1, 1], lw=THRESH_LW, color="k")
    ax.text(1, 3.5, "shuff\nthresh", fontsize=tick_label,\
            horizontalalignment='center')

    ax.set_xlim([0, 1.2])
    ax.set_xticks(np.arange(0, 1.4, 0.6))
    ax.set_yticks([0, 3, 6])
    ax.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax.spines["left"].set_bounds(0, 6)
    ax.set_ylabel("count", fontsize=axis_label, labelpad=1)

    return f, ax