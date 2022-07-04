import numpy as np
from sklearn.decomposition import PCA
from basic_analysis import tuning_curve_1d#, tuning_curve_2d

def cosine_sim(a, b):
    norm_a = np.linalg.norm(a, axis=0)
    norm_b = np.linalg.norm(b, axis=0)
    if len(a.shape) > 1:
        a_dot_b = np.zeros(a.shape[-1])
        for i in range(a.shape[-1]):
            a_dot_b[i] = a[:, i] @ b[:, i]
    else:
        a_dot_b = a @ b
    return a_dot_b / (norm_a * norm_b)

def proj_aB(v, P):
    # project vector v onto subspace P
    v_bar = P @ P.T @ v
    
    return cosine_sim(v_bar, v)
    

def remapping_dim(X0, X1):
    '''
    get the remapping dimension from two sets of 
    firing rates; shape (n_obs, hidden_size)
    '''
    X0_bar = np.mean(X0, axis=0)
    X1_bar = np.mean(X1, axis=0)
    return X0_bar - X1_bar

def position_subspace(X_tc, num_pcs=2):
    '''
    get the position subspace across contexts

    Params
    ------
    X_tc : ndarray, shape (n_maps, n_pos_bins, n_units)
        avg firing rate in each position bin for each context
    num_pcs : int
        n dims for the pos subspace
        use 2 for 1D model, 3 for 2D model

    Returns
    -------
    pos_subspace : ndarray, shape (n_units, n_pcs)
        basis set for the position subspace
    '''
    # check if tc is one or multiple maps
    if len(X_tc.shape)==2:
        X_tc = X_tc[None, :, :]
    n_maps = X_tc.shape[0]
    
    # mean center and normalize within each map
    for i in range(n_maps):
        X_map_tc = X_tc[i].copy()
        m = X_map_tc - np.mean(X_map_tc, axis=0, keepdims=True)
        m_norm = np.linalg.norm(m)
        m /= m_norm

        # concatenate
        if i == 0:
            m_all = m
        else:
            m_all = np.concatenate((m_all, m), axis=0)

    # PCA to get the position subspace (n_units, n_pcs)
    pca = PCA(n_components=num_pcs).fit(m_all)
    
    return pca.components_.T


def map_subspace(X0, pos0, \
                    num_pcs=2, model_2d=False):
    '''
    Get the subspace for one map

    Params
    ------
    X0 : ndarray, shape (n_obs, hidden_size)
        firing rates associated with a single map
    pos0 : ndarray, shape (n_obs)
        corresponding position estimates
    num_pcs : int
        n dims for the pos subspace
        use 2 for 1D model, 4 for 2D model
    model_2d : bool
        was the model trained on the 2D navigation task
    '''
    # position-binned firing rates (n_pos_bins, hidden_size)
    # if model_2d:
    #     tc_X0 = tuning_curve_2d(X0, pos0[:, 0], pos0[:, 1])
    #     X0_tc = tc_X0.reshape(-1, hidden_size)
    # else:
    X0_tc, _ = tuning_curve_1d(X0, pos0, n_pos_bins=250)

    # mean-center, normalize
    m1 = X0_tc - np.mean(X0_tc, axis=0, keepdims=True)
    m1_norm = m1 / np.linalg.norm(m1)

    # PCA to get the position subspace (hidden_size, num_pcs) (p1, p2)
    pca = PCA(n_components=num_pcs).fit(m1_norm)
    return pca.components_.T