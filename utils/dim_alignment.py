import numpy as np
from sklearn.decomposition import PCA
# from procrustes import tuning_curve_1d, tuning_curve_2d

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

# def position_subspace(X, pos_targ, map_targ, \
#                         num_pcs=2, model_2d=False):
#     '''
#     get the position subspace for each context
#     num_pcs : int
#         n dims for the pos subspace
#         use 2 for 1D model, 4 for 2D model
#     model_2d : bool
#         was the model trained on the 2D navigation task
#     '''
#     # split by context
#     X0 = X[map_targ==0]
#     X1 = X[map_targ==1]
#     pos0 = pos_targ[map_targ==0]
#     pos1 = pos_targ[map_targ==1]

#     # position-binned firing rates (n_pos_bins, hidden_size)
#     if model_2d:
#         tc_X0 = tuning_curve_2d(X0, pos0[:, 0], pos0[:, 1])
#         tc_X1 = tuning_curve_2d(X1, pos1[:, 0], pos1[:, 1])
#         X0_tc = tc_X0.reshape(-1, hidden_size)
#         X1_tc = tc_X1.reshape(-1, hidden_size)
#     else:
#         X0_tc = tuning_curve_1d(X0, pos0, n_bins=250)
#         X1_tc = tuning_curve_1d(X1, pos1, n_bins=250)

#     # mean-center, normalize, combine
#     m1 = X0_tc - np.mean(X0_tc, axis=0, keepdims=True)
#     m2 = X1_tc - np.mean(X1_tc, axis=0, keepdims=True)
#     m1_norm = np.linalg.norm(m1)
#     m2_norm = np.linalg.norm(m2)
#     m1 /= m1_norm
#     m2 /= m2_norm
#     X_all = np.concatenate((m1, m2), axis=0)

#     # PCA to get the position subspace (hidden_size, num_pcs) (p1, p2)
#     pca = PCA(n_components=num_pcs).fit(X_all)
#     return pca.components_.T


def map_subspace(X0, pos0, \
                    num_pcs=2, model_2d=False):
    '''
    get the subspace for one map
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
    if model_2d:
        tc_X0 = tuning_curve_2d(X0, pos0[:, 0], pos0[:, 1])
        X0_tc = tc_X0.reshape(-1, hidden_size)
    else:
        X0_tc = tuning_curve_1d(X0, pos0, n_bins=250)

    # mean-center, normalize
    m1 = X0_tc - np.mean(X0_tc, axis=0, keepdims=True)
    m1_norm = m1 / np.linalg.norm(m1)

    # PCA to get the position subspace (hidden_size, num_pcs) (p1, p2)
    pca = PCA(n_components=num_pcs).fit(m1_norm)
    return pca.components_.T