from plots import plot_trial, plot_rings, plot_contexts
import numpy as np
import torch
import matplotlib.pyplot as plt
from task import generate_batch

seed = 1234

task_params = {
    "num_steps": 400,
    "num_maps": 2,
    "remap_rate": 0.02, # expect 2 remaps every 100 steps
    "velocity_drift_stddev": 0.1,
    "velocity_noise_stddev": 0.3,
    "remap_pulse_duration": 2,
}
num_points = 4000

model = torch.load("./saved_models/12_35/model_weights.pt")

fig, ax = plot_trial(model, np.random.RandomState(seed), **task_params);
fig.savefig("example_trial2.pdf")

fig, ax, sc = plot_rings(
    model, np.random.RandomState(seed), num_points, **task_params)

fig, ax, sc = plot_contexts(model, np.random.RandomState(seed), num_points, **task_params)


d = np.squeeze(np.diff(model.readout_layer_map.weight.detach().numpy(), axis=0))
idx = np.argsort(d)
J = model.linear_hh.weight.detach().numpy()

fig, ax = plt.subplots(1, 1)
ax.imshow(J[idx][:, idx], interpolation="none", aspect="auto", clim=[-.2, .2])


a = model.readout_layer_pos.weight.detach().numpy()[0]
b = model.readout_layer_pos.weight.detach().numpy()[1]
idx = np.argsort(np.arctan2(a, b))
fig, ax = plt.subplots(1, 1)
ax.imshow(J[idx][:, idx], interpolation="none", aspect="auto", clim=[-.2, .2])

