import sys
sys.path.append("../utils/")
import numpy as np
from task import generate_batch
from task_2d import generate_batch as generate_batch_2d

from single_unit_analysis import get_FR

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from plot_utils import ring_colormap

from scipy.special import softmax
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


def plot_rings(model, random_state, num_points, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch(10, random_state, **kwargs)

    _, _, hidden_states = model(inp_init, inp_remaps, inp_vel)
 
    X = hidden_states.detach().numpy()
    X = X.reshape(-1, X.shape[-1])
    targ = pos_targets.detach().numpy().ravel()
    targ = (targ + np.pi) % (2 * np.pi) - np.pi

    pcs = PCA(n_components=3).fit_transform(X)
    idx = np.random.choice(targ.size, size=num_points, replace=False)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(
        *pcs[idx].T, c=targ[idx], cmap=ring_colormap(),
        lw=0, alpha=1, s=5
    )

    return fig, ax, sc


def plot_rings_2d(model, random_state, num_points, **kwargs):
    # get the neural activity for some example trials
    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch_2d(10, random_state, **kwargs)
    _, _, hidden_states = model(inp_init, inp_remaps, inp_vel)

    # reshape to (batch_size * num_steps, hidden_size)
    X = hidden_states.detach().numpy()
    X = X.reshape(-1, X.shape[-1])

    # get the positions and reshape (batch_size * num_steps, 2)
    targ = pos_targets.detach().numpy()
    targ = targ.reshape(-1, 2)
    targ = (targ + np.pi) % (2 * np.pi) - np.pi

    # perform PCA to get the 3D activity space
    pcs = PCA(n_components=3).fit_transform(X)
    idx = np.random.choice(X.shape[0], size=num_points, replace=False)

    # plot in PC space
    # color by position in 2D space
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(
        *pcs[idx].T, c=targ[idx, 0], cmap=ring_colormap(),
        lw=0, alpha=0.6, s=5
    )

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(
        *pcs[idx].T, c=targ[idx, 1], cmap=ring_colormap(),
        lw=0, alpha=0.6, s=5
    )

    return fig, ax, sc


def plot_fixed_pts(model, fixed_pts, random_state, num_points, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch(10, random_state, **kwargs)

    _, _, hidden_states = model(inp_init, inp_remaps, inp_vel)
 
    # find the top 3 PCs for the neural activity space
    X = hidden_states.detach().numpy()
    X = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=3).fit(X)

    # project into that space
    pcs_X = pca.transform(X)
    pcs_fixed_pts = pca.transform(fixed_pts)

    # identify the position at each time step (for colormap)
    targ = pos_targets.detach().numpy().ravel()
    targ = (targ + np.pi) % (2 * np.pi) - np.pi

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # plot the neural trajectories (subsampled)
    idx = np.random.choice(targ.size, size=num_points, replace=False)
    sc = ax.scatter(
        *pcs_X[idx].T, c=targ[idx], cmap=ring_colormap(),
        lw=0, alpha=1, s=2
    )

    # plot the fixed points
    sc = ax.scatter(
        *pcs_fixed_pts.T, c='k',
        lw=0, alpha=1, s=10
    )

    return fig, ax, sc


def plot_fixed_pts_2d(model, fixed_pts, random_state, num_points, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch_2d(10, random_state, **kwargs)

    _, _, hidden_states = model(inp_init, inp_remaps, inp_vel)
 
    # find the top 3 PCs for the neural activity space
    X = hidden_states.detach().numpy()
    X = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=3).fit(X)

    # project into that space
    pcs_X = pca.transform(X)
    pcs_fixed_pts = pca.transform(fixed_pts)

    # get the positions and reshape (batch_size * num_steps, 2)
    targ = pos_targets.detach().numpy()
    targ = targ.reshape(-1, 2)
    targ = (targ + np.pi) % (2 * np.pi) - np.pi

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # plot the neural trajectories (subsampled)
    idx = np.random.choice(X.shape[0], size=num_points, replace=False)
    sc = ax.scatter(
        *pcs_X[idx].T, c=targ[idx, 1], cmap=ring_colormap(),
        lw=0, alpha=0.6, s=2
    )

    # plot the fixed points
    sc = ax.scatter(
        *pcs_fixed_pts.T, c='k',
        lw=0, alpha=0.8, s=10
    )

    return fig, ax, sc


def plot_contexts(model, random_state, num_points, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch(10, random_state, **kwargs)

    _, _, hidden_states = model(inp_init, inp_remaps, inp_vel)
 
    X = hidden_states.detach().numpy()
    X = X.reshape(-1, X.shape[-1])
    targ = map_targets.detach().numpy().ravel()
    colors = {
        0: "k",
        1: "g",
        2: "r",
    }

    pcs = PCA(n_components=3).fit_transform(X)
    idx = np.random.choice(targ.size, size=num_points, replace=False)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sc = ax.scatter(
        *pcs[idx].T, c=[colors[t] for t in targ[idx]],
        lw=0, alpha=1, s=5, 
    )

    return fig, ax, sc

def plot_init_pos_perf(model, random_state, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch(124, random_state, **kwargs)

    pos_outputs, _, _ = model(inp_init, inp_remaps, inp_vel)
    pos_outputs = pos_outputs.detach().numpy()

    pred = np.arctan2(pos_outputs[0, :, 1], pos_outputs[0, :, 0])
    targ = pos_targets.detach().numpy()[0, :, 0]

    fig, ax = plt.subplots(1, 1)
    ax.scatter(pred, targ, lw=0, color="k", s=20)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], color="r")

    fig, ax = plt.subplots(1, 1)

    for (x1, y1), t in zip(pos_outputs[1], targ):
        x2, y2 = np.cos(t), np.sin(t)
        plt.plot([x1, x2], [y1, y2], color="k")
        plt.scatter(x1, y1, color="r", lw=0, s=20)
        plt.scatter(x2, y2, color="b", lw=0, s=20)


def plot_trial(model, random_state, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch(1, random_state, **kwargs)

    pos_outputs, map_logits, states = model(inp_init, inp_remaps, inp_vel)

    pos_outputs = pos_outputs.detach().numpy()[:, 0, :]
    pred = np.arctan2(pos_outputs[:, 1], pos_outputs[:, 0])
    targ = pos_targets.detach().numpy()[:, 0, :]

    targ = (targ + np.pi) % (2 * np.pi) - np.pi

    fig, axes = plt.subplots(5, 1, \
        gridspec_kw=dict(height_ratios=[2, 2, 1, 1, 1]), \
        sharex=True)
    axes[0].plot(pred, label="pred")
    axes[0].plot(targ, label="targ")
    axes[0].legend(title="position")
    axes[0].set_ylim(-np.pi - .2, np.pi + .2)
    axes[0].axhline(-np.pi, color="k", dashes=[2, 2], alpha=.5)
    axes[0].axhline(np.pi, color="k", dashes=[2, 2], alpha=.5)

    axes[1].plot(inp_vel[:, 0])

    map_logits = map_logits.detach().numpy()[:, 0, :]
    inp_remaps = inp_remaps.detach().numpy()[:, 0, :]

    # context prediction
    axes[2].plot(softmax(map_logits, axis=1)[:, 0], color="red", label="A")
    axes[2].plot(softmax(map_logits, axis=1)[:, 1], color="black", label="B")
    # axes[2].plot(softmax(map_logits, axis=1)[:, 2], color="green", label="C")
    axes[2].legend(title="context pred")

    # context input
    axes[3].plot(inp_remaps[:, 0], color="red", label="A")
    axes[3].plot(inp_remaps[:, 1], color="black", label="B")
    # axes[3].plot(inp_remaps[:, 2], color="green", label="C")
    axes[3].legend(title="context cues")

    # ground truth
    axes[4].plot(np.abs(map_targets-1), color="red", label="A")
    axes[4].plot(map_targets, color="black", label="B")
    # axes[4].plot(inp_remaps[:, 2], color="green", label="C")
    axes[4].legend(title="ground truth")

    return fig, axes


def plot_init_pos_perf_2d(model, random_state, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch_2d(124, random_state, **kwargs)

    pos_outputs, _, _ = model(inp_init, inp_remaps, inp_vel)
    pos_outputs = pos_outputs.detach().numpy()

    pred_1 = np.arctan2(pos_outputs[0, :, 1], pos_outputs[0, :, 0])
    targ_1 = pos_targets.detach().numpy()[0, :, 0]
    pred_2 = np.arctan2(pos_outputs[0, :, 3], pos_outputs[0, :, 2])
    targ_2 = pos_targets.detach().numpy()[0, :, 1]

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].scatter(pred_1, targ_1, lw=0, color="k", s=20)
    ax[0].plot([-np.pi, np.pi], [-np.pi, np.pi], color="r")
    ax[1].scatter(pred_2, targ_2, lw=0, color="k", s=20)
    ax[1].plot([-np.pi, np.pi], [-np.pi, np.pi], color="r")

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    for (x1, y1), t in zip(pos_outputs[0, :, :2], targ_1):
        x2, y2 = np.cos(t), np.sin(t)
        ax[0].plot([x1, x2], [y1, y2], color="k")
        ax[0].scatter(x1, y1, color="r", lw=0, s=20)
        ax[0].scatter(x2, y2, color="b", lw=0, s=20)
    for (x1, y1), t in zip(pos_outputs[0, :, 2:], targ_2):
        x2, y2 = np.cos(t), np.sin(t)
        ax[1].plot([x1, x2], [y1, y2], color="k")
        ax[1].scatter(x1, y1, color="r", lw=0, s=20)
        ax[1].scatter(x2, y2, color="b", lw=0, s=20)


def plot_trial_2d(model, random_state, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch_2d(1, random_state, **kwargs)

    pos_outputs, map_logits, states = model(inp_init, inp_remaps, inp_vel)

    pos_outputs = pos_outputs.detach().numpy()[:, 0, :]
    pred_1 = np.arctan2(pos_outputs[:, 1], pos_outputs[:, 0])
    pred_2 = np.arctan2(pos_outputs[:, 3], pos_outputs[:, 2])

    targ = pos_targets.detach().numpy()[:, 0, :]
    targ = (targ + np.pi) % (2 * np.pi) - np.pi
    targ_1 = targ[:, 0]
    targ_2 = targ[:, 1]

    fig, axes = plt.subplots(6, 1, \
        gridspec_kw=dict(height_ratios=[2, 2, 2, 1, 1, 1]), \
        sharex=True)
    axes[0].plot(pred_1, label="pred_1")
    axes[0].plot(targ_1, label="targ_1")
    axes[0].legend(title="position")
    axes[0].set_ylim(-np.pi - .2, np.pi + .2)
    axes[0].axhline(-np.pi, color="k", dashes=[2, 2], alpha=.5)
    axes[0].axhline(np.pi, color="k", dashes=[2, 2], alpha=.5)

    axes[1].plot(pred_2, label="pred_2")
    axes[1].plot(targ_2, label="targ_2")
    axes[1].legend(title="position")
    axes[1].set_ylim(-np.pi - .2, np.pi + .2)
    axes[1].axhline(-np.pi, color="k", dashes=[2, 2], alpha=.5)
    axes[1].axhline(np.pi, color="k", dashes=[2, 2], alpha=.5)

    axes[2].plot(inp_vel[:, 0, :])

    map_logits = map_logits.detach().numpy()[:, 0, :]
    inp_remaps = inp_remaps.detach().numpy()[:, 0, :]

    # context prediction
    axes[3].plot(softmax(map_logits, axis=1)[:, 0], color="red", label="A")
    axes[3].plot(softmax(map_logits, axis=1)[:, 1], color="black", label="B")
    # axes[2].plot(softmax(map_logits, axis=1)[:, 2], color="green", label="C")
    axes[3].legend(title="context pred")

    # context input
    axes[4].plot(inp_remaps[:, 0], color="red", label="A")
    axes[4].plot(inp_remaps[:, 1], color="black", label="B")
    # axes[3].plot(inp_remaps[:, 2], color="green", label="C")
    axes[4].legend(title="context cues")

    # ground truth
    axes[5].plot(np.abs(map_targets-1), color="red", label="A")
    axes[5].plot(map_targets, color="black", label="B")
    # axes[4].plot(inp_remaps[:, 2], color="green", label="C")
    axes[5].legend(title="ground truth")

    return fig, axes


def plot_tuning_curves(model, random_state, batch_size, **kwargs):

    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch(batch_size, random_state, **kwargs)

    # get unit firing rates from the model
    # shape: n_steps x n_batch x hidden_size
    _, _, firing_rates = model(inp_init, inp_remaps, inp_vel)

    ''' reformat the data '''
    # convert to numpy arrays
    pos_targets = pos_targets.detach().numpy()
    map_targets = map_targets.detach().numpy()
    firing_rates = firing_rates.detach().numpy()

    # flatten everything
    map_targets = map_targets.reshape(-1)
    pos_targets = pos_targets.reshape(-1)
    # there is certainly a tidier way to do this
    # (need to make sure the dims match so data is not shuffled)
    for i in range(n_units):
        if i == 0:
            firing_rates_flat = firing_rates[:, :, i].reshape(-1)
        else:
            firing_rates_flat = np.row_stack((firing_rates_flat, firing_rates[:, :, i].reshape(-1)))
    firing_rates = firing_rates_flat

    # convert distance travelled to position on circular track (-pi to pi)
    pos_targets_circ = (pos_targets + np.pi) % (2*np.pi) - np.pi

    # normalize firing rate for each unit
    for i in range(n_units):
        fr = firing_rates[i, :]
        fr -= np.min(fr)
        fr /= np.max(fr)
        firing_rates[i, :] = fr

    ''' get tuning curve for each unit in each map '''
    # filter by map
    FR_map1 = firing_rates[:, map_targets==0]
    pos_map1 = pos_targets_circ[map_targets==0]
    FR_map2 = firing_rates[:, map_targets==1]
    pos_map2 = pos_targets_circ[map_targets==1]
    
    # compute the tuning curves
    bin_size = (2 * np.pi) / 80
    tcs_map1, sem_map1 = get_FR(pos_map1, FR_map1, bin_size)
    tcs_map2, sem_map2 = get_FR(pos_map2, FR_map2, bin_size)
    
    ''' plot all the tuning curves ''' 
    def plot_TC(binned_pos, FR, SEM, ax, LW_MEAN = 1, LW_SEM = 0.3, color='k'):
        ax.plot(binned_pos, FR, color=color, lw=LW_MEAN, alpha=0.9)
        ax.fill_between(binned_pos, FR + SEM, FR - SEM,
                         color=color, linewidth=LW_SEM, alpha=0.3)
    
    # define the fig size and shape
    n_rows = np.ceil(n_units/10).astype(int)
    gs = gridspec.GridSpec(n_rows, 10, hspace=0.2, wspace=0.2)
    fig = plt.figure(figsize=(10, n_rows)) 
    LW_MEAN = 1
    LW_SEM = 0.3
        
    binned_pos = np.linspace(-np.pi, np.pi, num=80)
    for i in range(n_units):
        # get axis index
        row = i // 10
        col = i % 10
        ax = plt.subplot(gs[row, col])
        
        # plot firing rate and sem for each map
        plot_TC(binned_pos, 
                tcs_map1[i, :], 
                sem_map1[i, :], 
                ax, color='k')
        plot_TC(binned_pos, 
                tcs_map2[i, :], 
                sem_map2[i, :], 
                ax, color='r')
            
        # set and label axes
        ax.set_ylim([0, 1])
        ax.set_xlim([-np.pi, np.pi])
        if col > 0:
            ax.tick_params(labelleft=False)
        elif row == n_rows//2:
            ax.set_ylabel('normalized firing rate')
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([0, 200, 400]) # arbitrarily call the track 400cm
        if row < n_rows-1:
            ax.tick_params(labelbottom=False)
        elif col == 5:
            ax.set_xlabel('position on track (cm)')
        ax.tick_params(which='major', labelsize=8, pad=1)

    return fig, gs
