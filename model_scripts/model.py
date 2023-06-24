import torch
from torch import nn
from torch.functional import F
from task import generate_batch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


class RemapManualRNN(nn.Module):
    def __init__(self, hidden_size, num_maps, nonlinearity, num_spatial_dimensions):
        """
        Parameters
        ----------
        hidden_size : int, number of neurons.
        num_maps : int, number of maps.
        nonlinearity : str, determines neural firing rate nonlinearity.
        """

        super(RemapManualRNN, self).__init__()

        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.num_maps = num_maps
        self.num_spatial_dimensions = num_spatial_dimensions

        if nonlinearity == "linear":
            self._f = nn.Identity()
        elif nonlinearity == "tanh":
            self._f = nn.Tanh()
        elif nonlinearity == "relu":
            self._f = nn.ReLU()
        else:
            raise ValueError("Nonlinearity not recognized.")

        self.linear_initialize = nn.Linear(2 * num_spatial_dimensions, hidden_size)

        self.linear_ih = nn.Linear(num_spatial_dimensions + num_maps, hidden_size, bias=False)
        self.linear_hh = nn.Linear(hidden_size, hidden_size)

        self.readout_layer_map = nn.Linear(hidden_size, num_maps)
        self.readout_layer_pos = nn.Linear(hidden_size, 2 * num_spatial_dimensions)

    def forward(self, inp_init, inp_vel, inp_remaps):
        """
        Args:
          inp_init : n_batch x 2
          inp_vel : n_steps x n_batch x num_spatial_dimensions
          inp_remaps : n_steps x n_batch x num_maps

        Returns:
          pos_outputs : n_steps x n_batch x num_spatial_dimensions
          map_logits : n_steps x n_batch x num_maps
        """

        assert inp_vel.shape[-1] == self.num_spatial_dimensions
        assert inp_remaps.shape[-1] == self.num_maps

        # Initial states, (n_batch, hidden_size).
        state = self.linear_initialize(inp_init)

        # RNN inputs, (n_steps, n_batch, (1 + num_maps)).
        inputs = torch.cat((inp_vel, inp_remaps), axis=-1)

        # RNN outputs.
        pos_outputs = torch.empty((inputs.shape[0], inputs.shape[1], 2 * self.num_spatial_dimensions))
        map_outputs = torch.empty((inputs.shape[0], inputs.shape[1], self.num_maps))

        # RNN hidden states, (n_steps, n_batch, hidden_size).
        hidden_states = torch.empty((inputs.shape[0], inputs.shape[1], state.shape[1]))

        # This for loop is sub-optimal, but gets the job done.
        for t, inp in enumerate(inputs):
            state = self._f(self.linear_hh(state) + self.linear_ih(inp))
            pos_outputs[t] = self.readout_layer_pos(state)
            map_outputs[t] = self.readout_layer_map(state)
            hidden_states[t] = state

        return pos_outputs, map_outputs, hidden_states


    def one_step(self, prev_state, inp_vel, inp_remaps, \
                        return_pos=False):
            """
            Args:
                prev_state : n_batch x hidden_size
                inp_vel : n_batch x 1
                inp_remaps : n_batch x num_maps

            Returns:
                pos_outputs : n_steps x n_batch x 2
                next_state : n_batch x hidden_size
            """

            inputs = torch.cat((inp_vel, inp_remaps), axis=-1)
            next_state = self._f(self.linear_hh(prev_state) + self.linear_ih(inputs))

            if return_pos:
                pos_outputs = self.readout_layer_pos(next_state)
                return next_state, pos_outputs
            else:
                return next_state
