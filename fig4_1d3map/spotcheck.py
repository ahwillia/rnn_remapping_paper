import sys
sys.path.insert(1, '../model_scripts/')
sys.path.insert(1, '../utils/')
import os
import json
from model import RemapManualRNN
from task import RemapTaskLoss, generate_batch
from debug_plots import plot_trial, plot_init_pos_perf
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

MODEL_PATH = "../data/saved_models/1d_3map/11_101"
model = torch.load(f"{MODEL_PATH}/model_weights.pt")
random_state = np.random.RandomState(123)

with open(f"{MODEL_PATH}/task_params.json", "r") as f:
    task_params = json.load(f)

criterion = RemapTaskLoss(alpha=0.5)

# Generate batch.
inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
generate_batch(
    128,
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
print("total_loss: ", total_loss)
print("pos_loss: ", pos_loss)
print("map_loss: ", map_loss)


fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(np.load(f"{MODEL_PATH}/pos_losses.npy"), label="position decoding")
ax.plot(np.load(f"{MODEL_PATH}/map_losses.npy"), label="context decoding")
ax.legend()
plot_trial(model, random_state, **task_params)
plot_init_pos_perf(model, random_state, **task_params)
plt.show()
