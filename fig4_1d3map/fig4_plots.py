import sys
sys.path.append("../utils/")
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from plot_utils import simple_cmap, ring_colormap
from model_utils import sample_rnn_data, format_rnn_data
from basic_analysis import tuning_curve_1d, compute_misalignment

from scipy.special import softmax
from sklearn.decomposition import PCA
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
map_colors = [c1, c2, c3]

# rnn position input/output colors
pos_col = 'xkcd:cobalt blue'
est_col = 'xkcd:saffron'


def plot_a1(inp_vel, inp_remaps):
    '''
    Inputs for the 1D navigation/working memory task.
    (3 context inputs)

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
    n_maps = inp_remaps.shape[-1]

    # fig params
    gs = gridspec.GridSpec(5, 1, hspace=0)
    f = plt.figure(figsize=(1, 1))
    VEL_LW = 0.8
    CUE_LW = 0.8

    # plot velocity input
    ax0 = plt.subplot(gs[0])
    ax0.plot(inp_vel,
                c=pos_col, lw=VEL_LW)
    ax0.set_title('velocity',
                      fontsize=axis_label, pad=5)
    ax0.set_axis_off()

    # white space
    ax1 = plt.subplot(gs[1])
    ax1.set_axis_off()

    # plot context cue inputs
    for i in range(n_maps):
        ax = plt.subplot(gs[i+2])
        if i == 0:
            ax.set_title('context',
                          fontsize=axis_label, pad=0)
        ax.plot(inp_remaps[:, i], 
                 c=map_colors[i], lw=CUE_LW)
        ax.set_ylim([-0.2, 1.5])
        ax.set_axis_off()

    return f, gs


def plot_a2(pos_targets, pos_outputs, map_logits):
    '''
    Outputs for the 1D navigation/working memory task.
    (3 contexts)

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
    n_maps = map_logits.shape[-1]

    # fig params
    gs = gridspec.GridSpec(5, 1, hspace=0)
    f = plt.figure(figsize=(1, 1))
    TRUE_POS_LW = 1
    EST_POS_LW = 0.8
    CTXT_LW = 0.8

    # plot pos output vs. true pos
    ax0 = plt.subplot(gs[0])
    ax0.plot(targ, c=pos_col,
             lw=TRUE_POS_LW, 
             zorder=0)
    ax0.plot(pred, c=est_col,
             dashes=[1, 1], lw=EST_POS_LW,
             zorder=1)
    ax0.set_title('position',
                      fontsize=axis_label, pad=4)
    xlims = ax0.get_xlim()
    ax0.hlines([np.pi, -np.pi], xlims[0], xlims[1], 
               colors='k', alpha=0.6,
               linestyles='dashed', lw=CTXT_LW)
    ax0.set_axis_off()

    # white space
    ax1 = plt.subplot(gs[1])
    ax1.set_axis_off()

    # plot context cue inputs
    for i in range(n_maps):
        ax = plt.subplot(gs[i+2])
        if i == 0:
            ax.set_title('context',
                         fontsize=axis_label, pad=0)
        ax.plot(softmax(map_logits, axis=1)[:, i],
                c=map_colors[i], lw=CTXT_LW)
        ax.set_ylim([-0.2, 1.5])
        ax.set_axis_off()

    return f, gs

def plot_b(X, map_targ, pos_targ, n_ex_units=6):
    '''
    Tuning curves split by context for example RNN units.

    Params
    ------
    X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    map_targs : ndarray, shape(n_obs,)
        true context at each observation
    pos_targs : ndarray, shape(n_obs,)
        true positions at each observation
    n_ex_units : int
        number of example units to plot
        must be even
    '''

    # data params
    n_units = X.shape[-1]
    n_maps = np.max(map_targ) + 1

    # get the position binned firing rates for each map
    fr_0, _ = tuning_curve_1d(X[map_targ==0],
                                pos_targ[map_targ==0],
                                smooth=True)
    fr_1, _ = tuning_curve_1d(X[map_targ==1],
                                pos_targ[map_targ==1],
                                smooth=True)
    fr_2, _ = tuning_curve_1d(X[map_targ==2],
                                pos_targ[map_targ==2], 
                                smooth=True)
    n_pos_bins = fr_0.shape[0]
    binned_pos = np.linspace(0, 2*np.pi, num=n_pos_bins)

    # normalize within each unit
    all_fr = np.stack((fr_0, fr_1, fr_2)).copy()
    all_fr -= np.min(all_fr, axis=(0, 1))[None, None, :]
    all_fr /= np.max(all_fr, axis=(0, 1))[None, None, :]

    # select example units
    possible_units = np.arange(n_units)
    possible_units = possible_units[np.max(np.mean(all_fr, axis=0), axis=0) > 0.3]
    ex_units = np.random.choice(possible_units, n_ex_units, replace=False)

    # fig params
    n_col = int(n_ex_units/2)
    gs = gridspec.GridSpec(2, n_col,\
                           hspace=0.3, wspace=0.2)
    f = plt.figure(figsize=(3, 1))
    TC_LW = 1.2

    # plot firing rate vs. position for each map
    for i, u in enumerate(ex_units):
        # set axis
        if i < n_col:
            ax = plt.subplot(gs[0, i])
        else:
            ax = plt.subplot(gs[1, i-n_col])
            
        for j in range(n_maps):
            ax.plot(binned_pos, all_fr[j, :, u],
                    map_colors[j], lw=TC_LW, alpha=1)
            
        # ticks and lims
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['bottom'].set_bounds(0, 2*np.pi)
        ax.set_xlim([0, 2*np.pi])
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_ylim([0, 1.2])
        ax.set_yticks([0, 0.5, 1])
        
        # labels
        ax.tick_params(which='major', labelsize=tick_label, pad=0.5)
        if i == 1:
            ax.set_title('example RNN units', fontsize=title_size, pad=3)
        if i == n_col:
            ax.set_ylabel('firing rate', verticalalignment='bottom', y=1.1,\
                            fontsize=axis_label, labelpad=0.5)
        elif i > 0:
            ax.tick_params(labelleft=False)
        if i < n_col:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels([0, '$\pi$', '2$\pi$'])        
        if i == (n_ex_units - 2):
            ax.set_xlabel('track position (rad.)',\
                            fontsize=axis_label, labelpad=1)

    return f, gs


def plot_c(X, pos_targets,
    num_points=1000,
    axlim=2,
    reflect_x=False,
    reflect_y=False,
    reflect_z=False):
    
    '''
    ** copied exactly from fig1_plots **

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


''' POSSIBLE SUPPLEMENTAL FIGS '''
def plot_supp_1(data_folder, model_IDs):
    '''
    ** copied exactly from fig1_plots **

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


def plot_supp_3(data_folder, model_IDs, \
                top_num=4, top_num_1=2, \
                most_var_thresh=0.90):
    '''
    Average variance explained by each component across maps.
    Prints results within vs. across manifolds.

    Params
    ------
    X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    map_targets : ndarray, shape(n_obs,)
        true context at each observation
    top_num : int
        number of PCs to check for all maps
    top_num_1 : int
        number of PCs to check for each map
    most_var_thresh : float
        threshold for how many components explain "most of the variance"
    '''
    # data params
    inputs, outputs, targets = sample_rnn_data(data_folder, model_IDs[0])
    X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                            targets["map_targets"],\
                                            targets["pos_targets"])
    n_models = len(model_IDs)
    n_maps = np.max(np.unique(map_targ)) + 1
    hidden_size = X.shape[-1]

    # figure params
    f = plt.figure(figsize=(1.5, 0.8))
    DOT_SIZE = 5
    DOT_LW = 0.2
    CUM_LW = 1

    all_var = np.zeros((n_models, hidden_size))
    all_var_maps = np.zeros((n_models, n_maps, hidden_size))
    for i, m_id in enumerate(model_IDs):
        # get the rnn data
        inputs, outputs, targets = sample_rnn_data(data_folder, m_id)
        X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                                targets["map_targets"],\
                                                targets["pos_targets"])
        hidden_size = X.shape[-1]

        # PCA on all manifolds
        pca = PCA().fit(X)
        var = pca.explained_variance_
        total_var = np.sum(var)
        pct_var = (var / total_var)
        cum_var = np.cumsum(pct_var)
        all_var[i, :] = cum_var

        # PCA on each manifold
        for j in range(n_maps):
            X_map = X[map_targ==j]
            pca_0 = PCA().fit(X_map)
            var_0 = pca_0.explained_variance_
            pct_var_0 = (var_0 / np.sum(var_0))
            cum_var_0 = np.cumsum(pct_var_0)
            all_var_maps[i, j, :] = cum_var_0

    # print the results
    avg_var = np.mean(all_var, axis=0)
    most_var = np.argmin(np.abs(avg_var - most_var_thresh))
    print(f'on average, {top_num} PCs explain {avg_var[top_num-1]:.2%} of the variance')
    print(f'on average, {most_var+1} PCs explain {avg_var[most_var]:.2%} of the variance')
    avg_var_maps = np.mean(all_var_maps, axis=(0, 1))
    print(f'on average, {top_num_1} PCs explain {avg_var_maps[top_num_1-1]:.2%} of the variance within each map')

    # plot the average across models
    avg_var = np.mean(all_var, axis=0)
    ax0 = plt.axes([0, 0, 1, 1])
    ax0.scatter(np.arange(hidden_size) + 1,
                avg_var,
                c='k', s=DOT_SIZE-2,
                lw=0, zorder=1)
    ax0.plot(np.arange(hidden_size) + 1, avg_var,
             c='k', lw=CUM_LW, zorder=0)

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
    ax0.set_title('1D position, 3 maps',
                  fontsize=title_size, pad=4)
    ax0.set_xlabel('dimension', fontsize=axis_label, labelpad=1)
    ax0.set_ylabel('variance\nexplained', 
                   fontsize=axis_label, labelpad=1)

    return f, ax0