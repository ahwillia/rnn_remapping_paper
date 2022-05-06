import os
import json
from model import RemapManualRNN
from plots import plot_trial, plot_init_pos_perf
from task import RemapTaskLoss, generate_batch
from torch.optim import SGD, Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import torch

'''
perform SGD over the hidden states to minimize the velocity
finds fixed and slow points in the system
'''
### HYPERPARAMETERS ###
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
}


### INPUTS ###
# use the trained model to get a batch of hidden states
inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
        generate_batch(1000, random_state, **task_params)
_, _, hidden_states = model(inp_init, inp_remaps, inp_vel)

# randomly sample initial guesses from the hidden states
num_steps, num_batch, hidden_size = hidden_states.size()
idx = np.random.choice(num_steps, size=num_batch, replace=True)
init_states = torch.zeros([num_batch, hidden_size])
for b, i in enumerate(idx):
    init_states[b, :] = hidden_states[i, b, :]

# initial state guesses (num_batch, hidden_size)
prev_states = torch.nn.Parameter(init_states)

# set the velocity and context inputs to zero
num_maps = task_params["num_maps"]
inp_vel = torch.zeros(num_batch, 1)
inp_remaps = torch.zeros(num_batch, num_maps)


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


### SAVE OUTPUTS ###
outdir = f"./saved_models/{model_ID}/"
np.save(outdir + "vel_losses.npy", vel_losses)
next_states_np = next_states.detach().numpy()
np.save(outdir + "states_fixed_pt.npy", next_states_np)

fig, ax = plt.subplots(1, 1, figsize=(6, 2))
ax.plot(vel_losses)
ax.set_title('velocity loss')
plt.show()