import sys
sys.path.append("../utils/")
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

from plot_utils import ring_colormap
from model_utils import sample_rnn_data, format_rnn_data
from basic_analysis import tuning_curve_1d, compute_misalignment
import fig2_align_analysis as rnn

from scipy.special import softmax
from sklearn.decomposition import PCA
from scipy import stats

# font sizes
title_size = 10
axis_label = 9
tick_label = 7


def plot_c(data_folder, model_IDs):
    '''
    Histogram showing the difference between the single remapping
    vector for perfectly aligned rings (v) and the true remapping
    vectors between each pair of position bins (xi_p).
    '''
    # data params
    n_pos_bins = 50
    n_shuffle = 100
    n_models = len(model_IDs)

    # get the geometry of the remap vectors xi_p
    d_xi = np.zeros([n_models, n_pos_bins])
    avg_Wxi = np.zeros([n_models, n_pos_bins])
    d_shuff = np.zeros([n_models, n_shuffle, n_pos_bins])
    for i, m_id in enumerate(model_IDs):
        print(f'model {i + 1} of {n_models}')
        all_vector_results = rnn.remap_vector_geometry(data_folder, m_id, \
                                                        n_pos_bins=n_pos_bins, \
                                                        n_shuffle=n_shuffle)  
        avg_Wxi[i], d_xi[i], d_shuff[i] = all_vector_results
    
    # angle between the remap vectors and position subspace
    print(f'mean W * xi = {np.mean(avg_Wxi):.3}')
    print(f's.e.m. W * xi = {stats.sem(avg_Wxi.ravel()):.3}')

    # fig params 
    f, ax = plt.subplots(1, 1, figsize=(2.5, 1.2))
    BAR_LW = 1
    THRESH_LW = 2

    # plot the data
    ax.hist(d_xi.ravel(), np.linspace(0.0, 1.0, 30),\
            color="gray", lw=BAR_LW, edgecolor="k")
    ax.set_xlabel("normalized $v - \\xi_p$", fontsize=axis_label, labelpad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plot the shuffle threshold
    shuff_thresh = np.percentile(d_shuff.ravel(), 2.5)
    ax.vlines(shuff_thresh, 0, 200, linestyles=(200, [1, 1]), 
              lw=THRESH_LW, color='k')

    # labels and lims
    ax.set_xlim([0, 1.5])
    ax.set_xticks(np.arange(0, 1.6, 0.5))
    ax.set_yticks([0, 150, 300])
    ax.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax.spines["left"].set_bounds(0, 300)
    ax.set_ylabel("count", fontsize=axis_label, labelpad=1)
    ax.text(shuff_thresh, 220, "shuff\nthresh",
            fontsize=axis_label, horizontalalignment='center')

    return f, ax


def plot_d_e(data_folder, model_ID):
    '''
    Plots showing the dimensionality of the remap vectors
    for an example model
    '''
    # get the remap vectors
    xi_p, _, _ = rnn.compute_remap_vectors(data_folder, model_ID)

    # data params
    n_dims = np.min(xi_p.shape)
    spacer = n_dims / 50
    pos_bins = np.linspace(0, 2*np.pi, n_dims)

    # PCA on remap vectors
    xi_bar = xi_p - np.mean(xi_p, axis=0)
    pca = PCA().fit(xi_bar)
    var = pca.explained_variance_
    total_var_remap = np.sum(var)
    pct_var = (var / total_var_remap)
    cum_var = np.cumsum(pct_var)
    x_, y_ = PCA(n_components=2).fit_transform(xi_bar).T

    # get the variance of the full network
    inputs, outputs, targets = sample_rnn_data(data_folder, model_ID)
    X, _, _ = format_rnn_data(outputs["hidden_states"],\
                                targets["map_targets"],\
                                targets["pos_targets"])
    X_bar = X - np.mean(X, axis=0)
    pca = PCA().fit(X_bar)
    var = pca.explained_variance_
    total_var_full = np.sum(var)

    # get the variance of the position subspaces
    pos_ring = rnn.compute_pos_ring(data_folder, model_ID)
    pos_bar = pos_ring - np.mean(pos_ring, axis=0)
    pca = PCA().fit(pos_bar)
    var = pca.explained_variance_
    total_var_pos = np.sum(var)

    # get the relative variance
    rel_var = np.asarray([
        total_var_remap / total_var_full,
        total_var_pos / total_var_full
    ])

    # fig params
    f = plt.figure(figsize=(2.1, 2.1))
    PC_LW = 1.5
    DOT_SIZE = 10
    map_col = 'xkcd:scarlet'
    pos_col = 'xkcd:cobalt blue'

    # plot the cumulative variance explained
    ax0 = plt.axes([0.1, 0.65, 0.4, 0.3])
    ax0.plot(np.arange(n_dims) + 1, cum_var, c='k', lw=PC_LW, zorder=0)
    ax0.scatter(np.arange(2) + 1,
                cum_var[:2],
                facecolors='r', edgecolors='k',
                s=DOT_SIZE, lw=0.5, zorder=1)
    ax0.set_ylim(0, 1.1)

    # ticks and labels
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['left'].set_bounds(0, 1)
    ax0.spines['bottom'].set_bounds(-spacer, n_dims)
    ax0.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax0.set_xlim((-spacer, n_dims+spacer))
    ax0.set_xticks([1, n_dims/2, n_dims])
    ax0.set_ylim((0, 1.05))
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_yticklabels([0, 0.5, 1])
    ax0.set_xlabel('dimension', fontsize=axis_label, labelpad=0)
    ax0.set_ylabel('cumulative\n var. explained', 
                   fontsize=axis_label, labelpad=1)

    # plot the relative variance compared to the position variance
    ax3 = plt.axes([0.75, 0.6, 0.15, 0.35])
    bar_colors = [map_col, pos_col]
    xcoords = [1, 2]
    ax3.bar(xcoords, rel_var,
        width=0.6, bottom=0,
        color=bar_colors, alpha=1, edgecolor='k')
    # ticks and lims
    ax3.set_xticks([])
    ax3.set_yticks(np.arange(0, 1.2, 0.5))
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_bounds(0, 1)
    ax3.spines['bottom'].set_bounds(xcoords[0] - 0.6,
                                      xcoords[-1] + 0.6)
    ax3.tick_params(labelbottom=False, which='major', 
                    labelsize=tick_label, pad=0.5)
    ax3.set_xlim([xcoords[0] - 0.7,
                    xcoords[-1] + 0.7])
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('relative var.', fontsize=axis_label, labelpad=1)

    # plot the remap vectors against the first 2 PCs
    ax1 = plt.axes([0, 0, 0.5, 0.5])
    ax1.scatter(x_, y_,
                c=pos_bins, cmap=ring_colormap(),
                alpha=1, lw=0, s=DOT_SIZE*1.5)

    # ticks and labels
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False, labelleft=False,
                    which='major', labelsize=tick_label, pad=0.5)
    xlims = ax1.get_xlim()
    ylims = ax1.get_ylim()
    ax1.set_xlim((xlims[0], 1.6))
    ax1.set_ylim((ylims[0], 1.6))
    ax1.spines['bottom'].set_bounds(xlims[0], 1)
    ax1.spines['left'].set_bounds(ylims[0], 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('PC 2', fontsize=axis_label, labelpad=3)
    ax1.set_ylabel('PC 1', fontsize=axis_label, labelpad=1) 

    # plot the covariance
    ax2 = plt.axes([0.7, 0.06, 0.3, 0.3])
    cov_xi = xi_p @ xi_p.T
    cov_xi = (cov_xi - np.min(cov_xi)) / np.max(cov_xi - np.min(cov_xi))
    ax2.imshow(cov_xi, clim=[0, 1], 
               aspect='auto', cmap='viridis')

    # ticks and labels
    ax2.tick_params(which='major', labelsize=tick_label, pad=0.5)
    ax2.set_xticks([0, (n_dims/2) - 1, n_dims-1])
    ax2.set_yticks([0, (n_dims/2) - 1, n_dims-1])
    ax2.set_xticklabels([1, n_dims//2, n_dims])
    ax2.set_yticklabels([1, n_dims//2, n_dims])
    ax2.set_xlabel('position bins', fontsize=axis_label, labelpad=1)
    ax2.set_ylabel('position bins', fontsize=axis_label, labelpad=1)
    ax2.set_title('covariance', fontsize=title_size, pad=3)

    return f