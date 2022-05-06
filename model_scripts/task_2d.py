from sklearn.utils import check_random_state
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d
import torch
from torch import nn
from torch.functional import F
from time import time


class RemapTaskLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Parameters
        ----------
        alpha : float
            Relative strength of context vs position loss. When
            alpha == 0, only decode position
            alpha == 1, only decode context
        """
        super(RemapTaskLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pos_outputs, map_logits, pos_targets, map_targets):
        """
        Parameters
        ----------
        pos_outputs : float, shape == (num_steps, batch_size, 4)
            Output of RNN, predicting position. The two output pairs specify
            cos(theta) and sin(theta) of the angular position in each of
            the two dimensions.

        map_logits : float, shape == (num_steps, batch_size, num_maps)
            Output of RNN, predicting log-probability of each context

        pos_targets : float, shape == (num_steps, batch_size, 2)
            Target position in radians (2D).

        map_targets : float, shape == (num_steps, batch_size)
            Integer targets for the context.

        Returns
        -------
        total_loss : float
            Total loss used for training.

        pos_loss : float
            Loss on predicting the angular position.

        map_loss : float
            Loss on predicting the context / map.
        """

        # Compute map decoding loss
        num_maps = map_logits.shape[-1]
        map_loss = self.ce_loss(
            map_logits.reshape(-1, num_maps),
            map_targets.reshape(-1)
        )

        # Flatten batch dimension
        pos_flat = pos_targets.reshape(-1, 2)      # shape == (batch_size * num_steps, 2)
        pos_out_flat = pos_outputs.reshape(-1, 4)  # shape == (batch_size * num_steps, 4)

        # Compute position decoding loss
        pos_loss = F.mse_loss(
            pos_out_flat,
            torch.cat((torch.cos(pos_flat[:, 0, None]), torch.sin(pos_flat[:, 0, None]),
                        torch.cos(pos_flat[:, 1, None]), torch.sin(pos_flat[:, 1, None])), 
                        axis=1)
        )

        # Compute total loss.
        total_loss = (1 - self.alpha) * pos_loss + self.alpha * map_loss

        # Return losses
        return total_loss, pos_loss, map_loss


def generate_batch(
        batch_size,
        random_state,
        **kwargs
    ):
    """
    Generates a batch of trials, 

    Parameters
    ----------
    batch_size : int, number of trials to generate
    random_state : int, seed for random numbers
    **kwargs : passed to generate_trial function

    Returns
    -------
    inp_init : torch.tensor, shape == (batch_size, 4)
    inp_remaps : torch.tensor, shape == (num_steps, batch_size, num_maps)
    inp_vel : torch.tensor, shape == (num_steps, batch_size, 2)
    pos_targets : torch.tensor, shape == (num_steps, batch_size, 2)
    map_targets : torch.tensor of ints, shape == (num_steps, batch_size)
    """

    inp_init = []
    inp_remaps = []
    inp_vel = []
    pos_targets = []
    map_targets = []
    rs = check_random_state(random_state)

    for k in range(batch_size):
        init, vel, remaps, pos, mp = generate_trial(rs, **kwargs)

        inp_init.append(init)
        inp_remaps.append(remaps)
        inp_vel.append(vel)
        pos_targets.append(pos)
        map_targets.append(mp)


    # (batch_size, 4)
    inp_init = torch.tensor(
        np.stack(inp_init), dtype=torch.float32
    )

    # (num_steps, batch_size, num_maps)
    inp_remaps = torch.tensor(
        np.stack(inp_remaps, axis=1), dtype=torch.float32
    )

    # (num_steps, batch_size, 2)
    inp_vel = torch.tensor(
        np.stack(inp_vel, axis=1), dtype=torch.float32
    )

    # (num_steps, batch_size, 2)
    pos_targets = torch.tensor(
        np.stack(pos_targets, axis=1), dtype=torch.float32
    )

    # (num_steps, batch_size)
    map_targets = torch.tensor(
        np.stack(map_targets, axis=1), dtype=torch.long
    )

    return inp_init, inp_remaps, inp_vel, pos_targets, map_targets


def generate_trial(
        random_state,
        task='working_memory',
        num_steps=300,
        num_maps=2,
        remap_rate=0.02,
        velocity_drift_stddev=0.01,
        velocity_noise_stddev=0.03,
        remap_pulse_duration=5,
        sigma=0.5
    ):
    """
    Parameters
    ----------
    random_state : int, seed for random numbers.
    task : string, specifies remap input.
    num_steps : int, trial length.
    remap_rate : float, positive number controlling number of remap events.
    velocity_drift_stddev : float, standard deviation of average velocity.
    velocity_noise_stddev : float, controls noise in the velocity.
    remap_pulse_duration : int, how long a remap signal lasts.
    sigma : float, standard deviation for Gaussian noise (for task='inference').

    Returns
    -------
    inp_init : length-4 vector, represents initial position on torus.
    inp_vel : matrix, shape (num_steps, 2), 2d velocity at each time step.
    inp_remaps : matrix, shape (num_steps, num_maps)
        if task = 'working memory' : one-hot remap inputs.
        if task = 'inference' : noisy version of map_targets (and inverse).
    pos_targets : matrix, shape (num_steps, 2), target 2d position in radians.
    map_targets : vector of ints, len == num_steps, which map the network is in.
    """

    # Seed random number generator.
    rs = check_random_state(random_state)

    # Sample number of remaps.
    possible_remap_times = np.arange(remap_pulse_duration, num_steps, remap_pulse_duration)
    num_remaps = np.clip(rs.poisson(num_steps * remap_rate), 0, len(possible_remap_times))

    # Sample initial position
    initial_position = rs.uniform(low=-np.pi, high=np.pi, size=2)
    inp_init = np.array([
        np.cos(initial_position[0]),
        np.sin(initial_position[0]),
        np.cos(initial_position[1]),
        np.sin(initial_position[1])
    ])

    # Sample velocity trace.
    v_mean = velocity_drift_stddev * rs.randn(1, 2)
    v_noise = velocity_noise_stddev * rs.randn(num_steps, 2)
    inp_vel = v_mean + v_noise

    # Compute position by integrating velocity.
    pos_targets = initial_position + np.cumsum(inp_vel, axis=0)

    # Generate sequence of map ids. Choose first map randomly.
    map_ids = np.zeros(num_remaps + 1, dtype='int32')
    map_ids[0] = rs.choice(np.arange(num_maps))
    for i in range(1, num_remaps + 1):

        # Must remap to a new / unique map.
        avail_maps = np.setdiff1d(np.arange(num_maps), map_ids[i - 1])

        # Pick new map randomly
        map_ids[i] = rs.choice(avail_maps)

    # Generate random remap times.
    remap_times = np.sort(
        rs.choice(
            possible_remap_times,
            replace=False,
            size=num_remaps
        )
    )

    # Construct map decoding targets.
    map_dwell_times = np.diff(np.concatenate((
        [0], remap_times, [num_steps]
    )))
    map_targets = []
    for m, L in zip(map_ids, map_dwell_times):
        map_targets.append(np.full(L, m))
    map_targets = np.concatenate(map_targets)

    # Construct remapping input
    if task == 'working_memory': # input pulses        
        # 1 input stream, 2 maps
        # inp_remaps = np.zeros(num_steps)        
        # for t, m in zip(remap_times, map_ids[1:]):
        #     if m==0:
        #         inp_remaps[t] = 1.0
        #     else:
        #         inp_remaps[t] = -1.0                

        # inp_remaps = maximum_filter1d(
        #     inp_remaps, remap_pulse_duration, axis=0
        # )
        
        # n-maps input streams, arbitrary n maps        
        # inp_remaps = np.random.normal(scale=sigma, size=(num_steps, num_maps))
        inp_remaps = np.zeros((num_steps, num_maps))
        for t, m in zip(np.concatenate(([0], remap_times)),\
                        map_ids):
            inp_remaps[t, m] = inp_remaps[t, m] + 1.0

        inp_remaps = maximum_filter1d(
            inp_remaps, remap_pulse_duration, axis=0
        )    
    elif task == 'inference':
        # 1 input stream, 2 maps
        # epsilon = np.random.normal(scale=sigma, size=(num_steps))
        # all_remap_times = np.concatenate(([0], remap_times, [-1]))
        # inp_remaps = np.zeros(num_steps)
        # for t0, t1, m in zip(all_remap_times[:-1],
        #                      all_remap_times[1:],
        #                      map_ids):
        #     if m==0:
        #         inp_remaps[t0:t1] = epsilon[t0:t1] + 1.0
        #     else:
        #         inp_remaps[t0:t1] = epsilon[t0:t1] - 1.0

        # n-maps input streams, arbitrary n maps
        epsilon = np.random.normal(scale=sigma, size=(num_steps, num_maps))
        all_remap_times = np.concatenate(([0], remap_times, [-1]))
        inp_remaps = np.zeros((num_steps, num_maps))
        for t0, t1, m in zip(all_remap_times[:-1],
                              all_remap_times[1:],
                              map_ids):
          inp_remaps[t0:t1, m] = epsilon[t0:t1, m] + 1.0
          inp_remaps[t0:t1, ~m] = epsilon[t0:t1, ~m] - 1.0
    else:
        print('task type not recognized!')
    # inp_remaps = inp_remaps[:, None]

    return inp_init, inp_vel, inp_remaps, pos_targets, map_targets
