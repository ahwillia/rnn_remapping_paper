import torch
import numpy as np

def extract_jacobian_general(func, x):
    """
    Compute Jacobian of arbitrary function `func` at
    point `x`. We assume that `x` is a vector and
    `func(x)` is also a vector.
    """

    # We need to track gradients / differentiate with
    # respect to x.
    x.requires_grad_(True)

    # Define space for the Jacobian.
    num_outputs = func(x).size()[0]
    num_inputs = x.size()[0]
    J = torch.empty((num_outputs, num_inputs))

    for i in range(num_outputs):

        # Define the function we want to differentiate
        # locally at input x.
        y = func(x)

        # Create a one-hot vector at position i.
        z = torch.zeros(num_inputs)
        z[i] = 1

        # Compute (z.T @ J) where J is the jacobian of
        # the function we care about at x.
        y.backward(z)

        # For this choice of z, the backward pass yields
        # the i-th row of the Jacobian matrix.
        J[i] = x.grad

        # Reset the gradient of x to zero.
        x.grad.zero_()

    return J

def extract_jacobian(model, x, **kwargs):
    """
    Compute Jacobian of a fixed point, `x`, by passing it through
    one step of the model, with 0 velocity or context input.
    """

    # We need to track gradients / differentiate with
    # respect to x.
    x = x.detach()
    x = x.detach()    
    x.requires_grad_(True)

    # Define space for the Jacobian.
    x_1 = model.one_step(prev_state=x, **kwargs)
    num_outputs = x_1.size()[0]
    num_inputs = x.size()[0]
    J = torch.empty((num_outputs, num_inputs))

    for i in range(num_outputs):

        # Define the function we want to differentiate locally at input x.
        y = model.one_step(prev_state=x, **kwargs)

        # Create a one-hot vector at position i.
        z = torch.zeros(num_inputs)
        z[i] = 1

        # Compute (z.T @ J)
        # where J is the jacobian of the function we care about at x.
        y.backward(z)

        # For this choice of z, the backward pass yields
        # the i-th row of the Jacobian matrix.
        J[i] = x.grad

        # Reset the gradient of x to zero.
        x.grad.zero_()

    return J


def dist_to_map(X1, X2, y):
    '''
    Determine where each element of y falls 
    along the remapping dimension.

    Params
    ------
    X1, X2 : ndarray, shape (n_obs, hidden_size)
        neural firing in each map
    y : ndarray, shape (n_pts, hidden_size)
        points in activity space

    Returns
    -------
    y_dist : ndarray, shape (n_pts,)
        distance along the remapping dim to each map
        -1 = in map 1; 1 = in map 2; 0 = between the maps
    '''

    # define the remapping dimension (hidden_size,)
    X1_bar = np.mean(X1, axis=0)
    X2_bar = np.mean(X2, axis=0)
    remap_dim = (X1_bar - X2_bar) / np.linalg.norm(X1_bar - X2_bar)

    # project onto the remapping dim
    proj_m1 = X1_bar @ remap_dim # map 1
    proj_m2 = X2_bar @ remap_dim # map 2
    proj_y = y @ remap_dim # (n_pts,)

    # get distance to map
    y_dist = (proj_y - proj_m1) / (proj_m2 - proj_m1)
    return 2 * (y_dist - .5) # classify -1 or 1