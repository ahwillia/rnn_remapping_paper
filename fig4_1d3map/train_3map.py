import os
import sys
import json
sys.path.insert(1, '../model_scripts/')
sys.path.insert(1, '../utils/')
from model import RemapManualRNN
from task import RemapTaskLoss, generate_batch
from torch.optim import SGD, Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

###   PARSE RANDOM SEEDS   ###
parser = argparse.ArgumentParser()
parser.add_argument(
    "--NPSEED",
    help="random seed for numpy",
    default=999,
    type=int
)
parser.add_argument(
    "--TORCHSEED",
    help="random seed for pytorch",
    default=998,
    type=int
)
args = parser.parse_args()

###   HYPERPARAMETERS   ###
random_state = np.random.RandomState(args.NPSEED)
torch.manual_seed(args.TORCHSEED)

task_params = {
    "num_steps": 1,
    "num_maps": 3,
    "num_spatial_dimensions": 1,
    "remap_rate": 0.02, # expect 2 remaps every 100 steps
    "velocity_drift_stddev": 0.1,
    "velocity_noise_stddev": 0.3,
    "remap_pulse_duration": 2,
}

train_params = {
    "batch_size": 124,
    "num_iters": 30000,
    "init_lr": 0.1,
    "lr_step_size": 50,
    "lr_step_gamma": 0.99,
    "momentum": 0.0,
    "grad_clip_norm": 2.0,
    "updates_per_difficulty_increase": 100,
    "difficulty_increase": 1,
}

rnn_params = {
    "nonlinearity": "relu",
    "hidden_size": 248,
    "num_maps": task_params["num_maps"],
    "num_spatial_dimensions": task_params["num_spatial_dimensions"],
}


criterion = RemapTaskLoss(alpha=0.5)
model = RemapManualRNN(**rnn_params)

optimizer = SGD(
    model.parameters(),
    lr=train_params["init_lr"],
    momentum=train_params["momentum"]
)
scheduler = StepLR(
    optimizer,
    step_size=train_params["lr_step_size"],
    gamma=train_params["lr_step_gamma"]
)


pos_losses, map_losses, grad_norms = [], [], []

for itercount in trange(train_params["num_iters"]):

    # Increase trial length.
    if (itercount % train_params["updates_per_difficulty_increase"]) == 0:
        task_params["num_steps"] += train_params["difficulty_increase"]

    # Prepare optimizer.
    optimizer.zero_grad()

    # Generate batch.
    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
    generate_batch(
        train_params["batch_size"],
        random_state,
        **task_params
    )

    # Forward pass.
    pos_outputs, map_logits, states = model(inp_init, inp_vel, inp_remaps)

    # Evaluate loss.
    total_loss, pos_loss, map_loss = \
    criterion(
        pos_outputs,
        map_logits,
        pos_targets,
        map_targets
    )

    pos_losses.append(pos_loss.item())
    map_losses.append(map_loss.item())

    # Compute and clip gradients.
    total_loss.backward()
    clip_grad_norm_(model.parameters(), train_params["grad_clip_norm"])

    # Compute and store the gradient norms.
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    grad_norms.append(total_norm ** (1. / 2))

    # Update parameters
    optimizer.step()
    scheduler.step()

outdir = f"../data/saved_models/1d_3map/{args.NPSEED}_{args.TORCHSEED}/"
os.makedirs(outdir, exist_ok=True)

# Save weights and loss curves.
torch.save(model, outdir + "model_weights.pt")
np.save(outdir + "pos_losses.npy", pos_losses)
np.save(outdir + "map_losses.npy", map_losses)

with open(outdir + "task_params.json", "w") as f:
    json.dump(task_params, f, indent=4, sort_keys=True)

with open(outdir + "train_params.json", "w") as f:
    json.dump(train_params, f, indent=4, sort_keys=True)

with open(outdir + "rnn_params.json", "w") as f:
    json.dump(rnn_params, f, indent=4, sort_keys=True)


# PLOTS -- used only for debugging.
# from debug_plots import plot_trial, plot_init_pos_perf
# fig, axes = plt.subplots(2, 1, sharex=True)
# axes[0].plot(pos_losses, label="position decoding")
# axes[0].plot(map_losses, label="context decoding")
# axes[0].legend()
# axes[1].plot(grad_norms)
# plot_trial(model, random_state, **task_params)
# plot_init_pos_perf(model, random_state, **task_params)
# plt.show()
