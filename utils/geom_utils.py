"""
Geometry utils. Make and apply rotations, e.g.
"""


from scipy.stats import ortho_group
from sklearn.utils.validation import check_random_state


def two_d_rotation(theta):
    """
    Returns 2 x 2 rotation matrix Q, which acts on row vectors, v @ Q.
    """
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])


def rand_rotate_around_center(X):
    """
    Return randomly rotated matrix X.
    """
    m, n = X.shape
    return rotate_around_center(X, rand_orth(n))


def rotate_around_center(X, Q):
    """
    For (m x n) matrix holding m points in n-d space, apply 
    rotation held in (n x n) matrix Q relative to the center
    of mass of X.
    """
    m = np.mean(X, axis=0)
    return ((X - m[None, :]) @ Q) + m[None, :]


def rand_orth(m, n=None, random_state=None):
    """
    Creates a random matrix with orthogonal columns or rows.
    Parameters
    ----------
    m : int
        First dimension
    n : int
        Second dimension (if None, matrix is m x m)
    random_state : int or np.random.RandomState
        Random number specification.
    Returns
    -------
    Q : ndarray
        An m x n random matrix. If m > n, the columns are orthonormal.
        If m < n, the rows are orthonormal. If m == n, the result is
        an orthogonal matrix.
    """
    rs = check_random_state(random_state)
    n = m if n is None else n

    Q = ortho_group.rvs(max(m, n), random_state=rs)

    if Q.shape[0] > m:
        Q = Q[:m]
    if Q.shape[1] > n:
        Q = Q[:, :n]

    return Q
