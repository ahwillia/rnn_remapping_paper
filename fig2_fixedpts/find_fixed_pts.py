import sys
import os
import json
sys.path.append("../utils/")

from model_utils import load_model_params, sample_rnn_data, format_rnn_data

from scipy.special import softmax
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from scipy.ndimage import maximum_filter1d

import torch
from torch import nn
from torch.functional import F
from torch.optim import SGD, Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

from time import time
from tqdm import trange

'''
perform SGD over the hidden states to minimize the velocity
finds fixed and slow points in the system
'''
# file paths
data_folder = f"../data/saved_models/1d_2map/"
outdir = f"../data/saved_models/1d_2map/{model_ID}/"

### HYPERPARAMETERS ###
model, task_params, _ = load_model_params(data_folder, model_ID)

train_params = {
    "batch_size": 124,
    "num_iters": 5000,
    "init_lr": 0.1,
    "lr_step_size": 50,
    "lr_step_gamma": 0.99,
    "momentum": 0.0,
    "grad_clip_norm": 2.0,
    "updates_per_difficulty_increase": 100,
    "difficulty_increase": 1,
    "random_sample": True
}

### INPUTS ###
# get sample rnn data
inputs, outputs, targets = sample_rnn_data(data_folder, model_ID)
X, map_targ, pos_targ = format_rnn_data(outputs["hidden_states"],\
                                                targets["map_targets"],\
                                                targets["pos_targets"])

# initial state guesses (num_points, hidden_size)
init_states = initial_pts(X)
init_states = torch.from_numpy(init_states)
prev_states = torch.nn.Parameter(init_states)

# set the velocity and context inputs to zero
inp_vel = torch.zeros(num_batch, task_params["num_spatial_dimensions"])
inp_remaps = torch.zeros(num_batch, task_params["num_maps"])

### SGD ###
# for stochastic gradient descent
optimizer = SGD(
    [prev_states],
    lr=train_params["init_lr"],
    momentum=train_params["momentum"]
)

# to update the learning rate
scheduler = StepLR(
    optimizer,
    step_size=train_params["lr_step_size"],
    gamma=train_params["lr_step_gamma"]
)


vel_losses = []

for itercount in trange(train_params["num_iters"]):

    # Prepare optimizer.
    optimizer.zero_grad()

    # Forward pass.
    next_states = model.one_step(prev_states, inp_vel, inp_remaps)

    # Evaluate loss - how close is the velocity to 0?
    vel_loss = torch.sum((prev_states - next_states)**2)
    vel_losses.append(vel_loss.item())

    # Compute gradient.
    vel_loss.backward()

    # Update parameters
    optimizer.step()
    scheduler.step()
    
    if itercount == (train_params["num_iters"] - 1):
        next_states, pos_outputs = model.one_step(prev_states, \
                                                    inp_vel, \
                                                    inp_remaps, \
                                                    return_pos=True)

### SAVE OUTPUTS ###
np.save(outdir + "vel_losses.npy", vel_losses)
fixed_pts_np = next_states.detach().numpy()
np.save(outdir + "states_fixed_pt.npy", fixed_pts_np)
pos_outputs_np = pos_outputs.detach().numpy()
np.save(outdir + "pos_fixed_pt.npy", pos_outputs_np)

### for debugging ###
## plot the velocity loss
# fig, ax = plt.subplots(1, 1, figsize=(6, 2))
# ax.plot(vel_losses)
# ax.set_title('velocity loss')
# plt.show()
#
## plot the fixed points in 3D PC space
# f, ax, _ = plot_fixed_pts(model,
#                           fixed_pts=fixed_pts,
#                           random_state=random_state,
#                           num_points=1000,
#                           **task_params
#                          )

# plt.show()


def initial_pts(X, num_pts=1000):
    '''
    Choose initialization points randomly from throughout the activity space.

    Params
    ------
    X : ndarray, shape (n_obs, n_units)
        RNN unit activity at each observation

    Returns
    -------
    init_states : ndarray, shape (num_pts, n_units)
        points in activity space from which to initialize
        the fixed point finder
    '''
    # find the top 3 PCs for the neural activity space
    pca = PCA(n_components=3).fit(X)
    pcs = pca.transform(X)

    # define the corners
    pc_max = np.max(pcs, axis=0)
    pc_min = np.min(pcs, axis=0)
    pc_corners = np.stack((pc_min, pc_max), axis=0)

    # randomly sample initial states
    init_pcs = torch.zeros([num_pts, 3])
    for x in range(3):
        x_min = np.min(pc_corners[:, x])
        x_max = np.max(pc_corners[:, x])    
        init_pcs[:, x] = (x_min - x_max) * torch.rand(num_pts) + x_max
        
    # transform back to full D
    init_states = pca.inverse_transform(init_pcs)

    return init_states


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