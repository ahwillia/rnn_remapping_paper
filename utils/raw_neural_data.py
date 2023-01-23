import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import scipy.io
import h5py
from pathlib import Path
from tqdm import trange

def load_np_data(data_folder, session_ID):
    """Loads the neuropixels matlab data struct and extracts relevant variables

    Returns
    -------
    data : dict
        dict containing behavioral and spiking data
        data['sp'] gives dict of spiking data

    """
    # load data
    path = f'{data_folder}{session_ID}/{session_ID}_data.mat'
    d = loadmat_sbx(path)

    if 'data' in d: # struct is from silicon probe data
        data = d['data']
        sp = data['sp']
        # print(sp.keys())
    else:
        print('could not recognize data format!')

    return sp


""" Scripts for Loading Matlab Structs """

def loadmat_sbx(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    print(filename)
    data_ = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data_)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes todict is called
    to change them to nested dictionaries
    """

    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])

    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_ca_mat(fname):
    """load results from cnmf"""

    ca_dat = {}
    try:
        with h5py.File(fname, 'r') as f:
            for k, v in f.items():
                try:
                    ca_dat[k] = np.array(v)
                except:
                    print(k + "not made into numpy array")
                    ca_dat[k] = v
    except:
        ca_dat = scipy.io.loadmat(fname)
        for key in ca_dat.keys():
            if isinstance(ca_dat[key], np.ndarray):
                ca_dat[key] = ca_dat[key].T
    return ca_dat