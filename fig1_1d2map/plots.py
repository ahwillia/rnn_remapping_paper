import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
''' general figure params '''
# font sizes
title_size = 10
axis_label = 9
tick_label = 7

# map colors
c1 = 'xkcd:scarlet'
c2 = 'k'

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
    cell_ID : ndaintrray
        ID number for the example cell.
    FR_0, FR_1 : ndarray
        firing rate by position within each map
        shape (n_pos_bins, n_cells)
    FR_0_sem, FR_1_sem : ndarray
        SEM for the firing rate arrays; shape (n_pos_bins, n_cells)
    binned_pos : ndarray
        centers of each position bin (cm); shape (n_pos_bins, )
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