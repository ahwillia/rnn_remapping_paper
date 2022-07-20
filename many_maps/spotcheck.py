import sys
sys.path.insert(1, '../model_scripts/')
sys.path.insert(1, '../utils/')
import os
import json
from model import RemapManualRNN
from debug_plots import plot_trial, plot_init_pos_perf
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

MODEL_PATH = "../data/saved_models/1d_2map/10_100"
model = torch.load(f"{MODEL_PATH}/model_weights.pt")
random_state = np.random.RandomState(123)

task_params = {
    "num_steps": 500,
    "num_maps": 2,
    "remap_rate": 0.02, # expect 2 remaps every 100 steps
    "velocity_drift_stddev": 0.1,
    "velocity_noise_stddev": 0.3,
    "remap_pulse_duration": 2,
}

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(np.load(f"{MODEL_PATH}/pos_losses.npy"), label="position decoding")
ax.plot(np.load(f"{MODEL_PATH}/map_losses.npy"), label="context decoding")
ax.legend()
plot_trial(model, random_state, **task_params)
plot_init_pos_perf(model, random_state, **task_params)
plt.show()
