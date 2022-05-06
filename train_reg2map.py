from model import RemapManualRNN
from task import RemapTaskLoss, generate_batch
from torch.optim import SGD, Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import torch

###   HYPERPARAMETERS   ###
random_state = np.random.RandomState(1234)
torch.manual_seed(1234)

task_params = {
    "num_steps": 1,
    "num_maps": 2,
    "remap_rate": 0.02, # expect 2 remaps every 100 steps
    "velocity_drift_stddev": 0.1,
    "velocity_noise_stddev": 0.3,
    "remap_pulse_duration": 5,
}

train_params = {
    "batch_size": 124,
    "num_iters": 20000,
    "init_lr": 0.1,
    "lr_step_size": 50,
    "lr_step_gamma": 0.99,
    "momentum": 0.0,
    "grad_clip_norm": 2.0
}

rnn_params = {
    "nonlinearity": "tanh",
    "hidden_size": 248,
    "num_maps": task_params["num_maps"]
}


criterion = RemapTaskLoss(lam_map=0.1, lam_ort=1.0, lam_par=1.0)
model = RemapManualRNN(**rnn_params)
with torch.no_grad():
    model.linear_hh.weight.mul_(0.05)

optimizer = SGD(
    model.parameters(),
    lr=train_params["init_lr"],
    momentum=train_params["momentum"],
    weight_decay=1e-2
)
# optimizer = Adam(
#     model.parameters(),
#     lr=train_params["init_lr"]
# )
scheduler = StepLR(
    optimizer,
    step_size=train_params["lr_step_size"],
    gamma=train_params["lr_step_gamma"]
)


pos_losses, map_losses, grad_norms = [], [], []

for itercount in trange(train_params["num_iters"]):

    if (itercount % 100) == 0:
        task_params["num_steps"] += 1

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
    pos_outputs, map_logits, states = model(inp_init, inp_remaps, inp_vel)

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

torch.save(model, "./saved_models/rnn_2map_weight_decay.pt")

# fig, axes = plt.subplots(2, 1, sharex=True)
# axes[0].plot(pos_losses, label="position decoding")
# axes[0].plot(map_losses, label="context decoding")
# axes[0].legend()
# axes[1].plot(grad_norms)

# plot_trial(model, random_state, **task_params)
# plot_init_pos_perf(model, random_state, **task_params); plt.show()

# plt.show()
