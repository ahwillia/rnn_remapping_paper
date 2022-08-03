import sys
sys.path.append("../utils/")
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from plot_utils import simple_cmap, ring_colormap
from model_utils import sample_rnn_data, format_rnn_data
from basic_analysis import tuning_curve_1d, compute_misalignment
import fig2_analysis as rnn

from scipy.special import softmax
from sklearn.decomposition import PCA
from scipy import stats


''' general figure params '''
# font sizes
title_size = 10
axis_label = 9
tick_label = 7

# colors
stable_col = 'xkcd:jungle green'
unstable_col = 'xkcd:gold'

def plot_a(X, fixed_pts, pos_targets, 
                ex_idx, not_ex_idx,
                plot_all_pts=True,
                num_points=1000,
                axlim=2,
                reflect_x=False,
                reflect_y=False,
                reflect_z=False):
    '''
    Fixed points with examples highlighted.

    Params
    ------
    X : ndarray, shape (n_obs, hidden_size)
        RNN unit activity at each observation
    fixed_pts : ndarray, shape (n_fixed_pts,)
        approximate fixed points
    pos_targets : ndarray, shape (n_obs,)
        true positions at each observation
    ex_idx : ndarray, shape (n_examples,)
        indices for the example fixed points
    not_ex_idx : ndarray, shape (n_examples,)
        indices for all the other fixed points
    plot_all_pts : bool
        plot all the fixed points
    '''
    # data params
    num_posbins, num_neurons = X.shape
    num_points = fixed_pts.shape[0]
    num_ex = ex_idx.shape[0]

    # mean center
    X_bar = X - np.mean(X, axis=0)

    # Find PCs
    pca = PCA(n_components=3)
    x_, y_, z_ = pca.fit_transform(X_bar).T
    x_fp, y_fp, z_fp = pca.transform(fixed_pts).T

    # Reflect axes if desired
    if reflect_x:
        x_ *= -1
        x_fp *= -1
    if reflect_y:
        y_ *= -1
        y_fp *= -1
    if reflect_z:
        z_ *= -1
        z_fp *= -1

    # fig params
    f = plt.figure(figsize=(2.5, 1.5))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')
    DOT_SIZE = 5
    PC_LW = 2

    COLORS = []
    for c in range(num_ex):
        if c < num_ex//2:
            COLORS.append(stable_col)
        else:
            COLORS.append(unstable_col)            

    if plot_all_pts:
        # plot the fixed points
        sc = ax.scatter(
            x_fp[not_ex_idx], y_fp[not_ex_idx], z_fp[not_ex_idx],
            c='xkcd:gray', lw=0,
            alpha=1, s=DOT_SIZE
        )

    # plot the examples
    sc = ax.scatter(
        x_fp[ex_idx], y_fp[ex_idx], z_fp[ex_idx],
        facecolors=COLORS, edgecolors='k',
        lw=0.5, alpha=1, s=DOT_SIZE+2
    )

    # plot shadow
    ax.scatter(
        x_fp, y_fp, 
        np.full(num_points, -axlim),
        color='k', alpha=.01, lw=0, s=DOT_SIZE)

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

    ax.set_title('fixed points',
                  fontsize=title_size, pad=-10)
    ax.view_init(azim=110, elev=30)
    ax.axis("off")

    return f, ax


def plot_b(max_eigs, ex_idx, tol):
    '''
    Plot the largest eigenvalue associated with each fixed point

    Indicate the tolerance for saddle vs. stable
    '''
    num_ex = ex_idx.shape[0]
    num_fixed_pts = max_eigs.shape[0]

    # fig params
    f, ax = plt.subplots(1, 1, figsize=(1, 1))
    DOT_SIZE=5
    LW=1

    COLORS = []
    for c in range(num_ex):
        if c < num_ex//2:
            COLORS.append(stable_col)
        else:
            COLORS.append(unstable_col)

    # plot fixed points
    xvals = np.arange(num_fixed_pts)
    ex_pts = max_eigs[ex_idx]
    ax.scatter(xvals + 1, max_eigs, 
               c='xkcd:gray', s=DOT_SIZE, lw=0,
               zorder=0)
    ax.scatter(ex_idx + 1, ex_pts, 
               facecolors=COLORS, edgecolors='k',
               s=DOT_SIZE+2, lw=0.5,
               zorder=1)

    # plot a line at 1
    xlims = ax.get_xlim()
    ax.hlines(1, xlims[0], xlims[1],
              linestyles='dashed',
              colors='k',
              lw=LW, zorder=2
             )

    # plot a lines to indicate saddle vs. stable cut-off
    ax.hlines([1-tol, 1+tol], 
          xlims[0], xlims[1],
          linestyles='dotted',
          colors='xkcd:deep red',
          lw=LW, zorder=2
         )

    # ticks and labels
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim([0.85, 1.45])
    ax.set_xlim([-40, num_fixed_pts+40])
    ax.set_yticks([1, 1.4])
    ax.set_xticks([1, num_fixed_pts//2, num_fixed_pts])
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.spines['left'].set_bounds(ylims[0], 1.4)
    ax.spines['bottom'].set_bounds(xlims[0], num_fixed_pts)

    ax.set_ylabel('max $R(\lambda)$', fontsize=axis_label, labelpad=1)
    ax.set_xlabel('fixed point', fontsize=axis_label, labelpad=1)
    ax.tick_params(which='major', labelsize=tick_label, pad=1)

    return f, ax


def plot_c(fp_dist, stable_idx, saddle_idx):
    '''
    Plot a histogram of fixed point projections onto the remapping dimension.

    Params
    ------
    fp_dist : ndarray, shape (n_fixed_pts,)
        projection of the fixed points onto the remapping dimension
        -1 = in map 1; 1 = in map 2; 0 = between the maps
    stable_idx : ndarray, shape (n_fixed_pts,)
        index for quasi-stable points
    saddle_idx : ndarray, shape (n_fixed_pts,)
        index for saddle points
    '''
    # fig params
    f, ax = plt.subplots(1, 1, figsize=(1.5, 0.9))
    bins = np.linspace(-1.1, 1.1, 40)

    # quasi-stable points
    ax.hist(fp_dist[stable_idx], bins=bins, 
            facecolor=stable_col, edgecolor='k',
            lw=1, alpha=1, label='stable'
           )

    # saddle points
    ax.hist(fp_dist[saddle_idx], bins=bins, 
            facecolor=unstable_col, edgecolor='k',
            lw=1, alpha=1, label='saddle'
           )

    # ticks and labels
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
        
    ax.tick_params(which='major', labelsize=tick_label, pad=1)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([-1, 0, 1])

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.spines['left'].set_bounds(ylims[0], 100)
    ax.spines['bottom'].set_bounds(xlims[0], xlims[1])

    ax.set_title('fixed pt. projections', fontsize=title_size, pad=2)
    ax.set_xlabel('remap dim', fontsize=axis_label, labelpad=1)

    return f, ax


def plot_d(Js, ex_idx):
    '''
    Plot the eigenvalues for example fixed points.

    Params
    ------
    Js : ndarray, shape (n_fixed_pts, hidden_size, hidden_size)
        Jacobian associated with each fixed point
    ex_idx : ndarray, shape (n_ex_pts,)
        index for each example
    '''
    num_ex = ex_idx.shape[0]

    # fig params
    gs = gridspec.GridSpec(1, 6, wspace=0.3, hspace=0.3)
    f = plt.figure(figsize=(5, 0.8))
    DOT_SIZE=5
    DOT_LW=0.5
    LW=1.5

    COLORS = []
    for c in range(num_ex):
        if c < num_ex//2:
            COLORS.append(stable_col)
        else:
            COLORS.append(unstable_col) 
                
    for i, ex in enumerate(ex_idx):
        ax = plt.subplot(gs[i])

        # plot the eigenvalues
        J = Js[ex]
        lam, V = np.linalg.eig(J)
        ax.scatter(lam.real, lam.imag,
                    facecolors=COLORS[i], edgecolors='k',
                    lw=DOT_LW, s=DOT_SIZE,
                    alpha=1, zorder=1)

        # plot unit circle
        circle = plt.Circle((0, 0), 1,
                            color='k', fill=False)
        ax.add_patch(circle)

        # labels, limits, etc.
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_bounds(-0.5, 0.5)
        ax.spines['bottom'].set_bounds(-0.5, 1.3)

        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([-0.5, 1.5])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xticks([0, 1.3])

        if (i == 1) | (i == 4):
            ax.set_title('ex. eigenvals', fontsize=title_size, pad=10)
            ax.set_xlabel('real($\lambda$)', fontsize=axis_label, labelpad=1)
        if i == 0:
            ax.set_ylabel('im($\lambda$)', fontsize=axis_label, labelpad=0)
        if i > 0:
            ax.tick_params(labelleft=False)
        ax.tick_params(which='major', labelsize=tick_label, pad=1)

    return f, gs