import sys
sys.path.append("../model_scripts/")
import os
import json
import torch
import numpy as np
from task import generate_batch

def load_model_params(data_folder, model_ID):
    model = torch.load(f"{data_folder}/{model_ID}/model_weights.pt")
    with open(f"{data_folder}/{model_ID}/task_params.json", 'r') as f:
        task_params = json.load(f)
    with open(f"{data_folder}/{model_ID}/rnn_params.json", 'r') as f:
        rnn_params = json.load(f)
    
    return model, task_params, rnn_params


def sample_rnn_data(data_folder, model_ID, batch_size=100):
    model, task_params, rnn_params =\
            load_model_params(data_folder, model_ID)

    # set random seeds
    NP_SEED = int(model_ID.split('_')[0])
    TORCH_SEED = int(model_ID.split('_')[1])
    random_state = np.random.RandomState(NP_SEED)
    torch.manual_seed(TORCH_SEED)

    # get sample neural activity, position targets, map targets, inputs
    inp_init, inp_remaps, inp_vel, pos_targets, map_targets = \
            generate_batch(batch_size, random_state, **task_params)

    pos_outputs, map_logits, hidden_states =\
            model(inp_init, inp_vel, inp_remaps)

    inputs = {
        "inp_init": inp_init, 
        "inp_remaps": inp_remaps,
        "inp_vel": inp_vel
    }

    outputs = {
        "pos_outputs": pos_outputs,
        "map_logits": map_logits,
        "hidden_states": hidden_states
    }

    targets = {
        "pos_targets": pos_targets,
        "map_targets": map_targets       
    }

    return inputs, outputs, targets


def format_rnn_data(hidden_states, map_targets, pos_targets):
    # unit activity
    X = hidden_states.detach().numpy()
    X = X.reshape(-1, X.shape[-1])

    # true contexts
    map_targ = map_targets.detach().numpy().ravel()

    # true positions
    pos_targ = pos_targets.detach().numpy()
    pos_targ = pos_targ.ravel()
    pos_targ = (pos_targ + np.pi) % (2 * np.pi) - np.pi

    return X, map_targ, pos_targ