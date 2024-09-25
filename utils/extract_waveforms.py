import os
import warnings
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
)


def list_sessions(data_folder):

    sessions = []

    for root, dirs, files in os.walk(data_folder, topdown=True):
        for d in list(dirs):
            if (d == ".phy") or (d == "video") or (d == "VR"):
                dirs.remove(d)
        for fname in files:
            if fname.endswith("MEC_cellIDs.npy"):
                sessions.append(root)

    return sorted(sessions)

def run_session(data_folder, session_index):
    # Determine path to session.
    datapath = list_sessions(data_folder)[session_index]

    # Determine session ID.
    session_id = os.path.split(datapath)[-1]
    session_id = session_id[:-3]

    logging.info(f"SESSIONS: {list_sessions(data_folder)}")
    logging.info(f"SESSION NAME  : {session_id}")
    logging.info(f"DATAPATH      : {datapath}")

    # Find path to neuropixel output files.
    imec_path = os.path.join(datapath, f"{session_id}_g0_imec0")

    # Delete trailing zero when necessary
    if not os.path.exists(imec_path):
        imec_path = imec_path[:-1]

    # Load epochs and cell ids
    cellids_file = os.path.join(datapath, grab_files(datapath, "MEC_cellIDs.npy"))
    # epochs_file = os.path.join(datapath, "epochs.npy")
    epochs_file = os.path.join(
        "/mnt/home/awilliams/code/rnn_remapping_paper/data/subset_neural_data",
        f"{session_id}_1_epochs.npy"
    )

    # Compute mean and stddev of waveforms in each epoch.
    mn, sd, cnts = extract_waveforms(
        imec_path,
        epoch_times=np.load(epochs_file),
        cell_ids=np.load(cellids_file)
    )

    # Save results
    np.save(os.path.join(datapath, f"{session_id}_mean_waveforms.npy"), mn)
    np.save(os.path.join(datapath, f"{session_id}_stddev_waveforms.npy"), sd)
    np.save(os.path.join(datapath, f"{session_id}_spike_counts.npy"), cnts)


def extract_waveforms(
        imec_path,
        epoch_times=None,
        cell_ids=None,
        spikes_per_epoch=100,
        pre_samples=None
    ):

    # Determine file path to neuropixel voltage traces and metadata.
    datafile = os.path.join(imec_path, grab_files(imec_path, ".ap.bin"))
    metafile = os.path.join(imec_path, grab_files(imec_path, ".ap.meta"))

    # Parse the metadata file.
    metadata = parse_metafile(metafile)

    # Extract the voltage traces.
    _raw_data = np.memmap(datafile, dtype='int16', mode='r')

    # Reshape to (num_timebins x num_channels).
    data_shape = (
        int(_raw_data.size / metadata['num_channels']), metadata['num_channels']
    )
    data = np.reshape(_raw_data, data_shape)

    # Load spike clusters and templates.
    ks_data = load_kilosort_data(
        imec_path,
        metadata['sample_rate'],
        convert_to_seconds = False
    )

    logging.info(f"min spike time: {ks_data['spike_times'].min()}")
    logging.info(f"max spike time: {ks_data['spike_times'].max()}")
    logging.info(f"sample rate: {metadata['sample_rate']}")

    # Load the epoch times and cell ids to extract.
    
    # By default, extract mean waveforms over the full session.
    if epoch_times is None:
        epoch_times = np.array([[0, data_shape[0]]])

    if cell_ids is None:
        cell_ids = ks_data["cluster_ids"]

    for i in cell_ids:
        if i not in ks_data["cluster_ids"]:
            raise ValueError(
                "Processing: {}\n\tcell id {} isn't in kilosort ids.")

    # Data dimensions
    num_units = len(cell_ids)
    num_epochs = len(epoch_times)
    num_channels = data.shape[1]
    timebins_per_spike = ks_data["unwhitened_temps"].shape[1]

    if pre_samples is None:
        pre_samples = timebins_per_spike // 2

    # Allocate space for mean and std dev. of waveforms
    mean_waveforms = np.full((
        num_epochs,
        num_units,
        num_channels,
        timebins_per_spike
    ), np.nan)
    stddev_waveforms = np.full((
        num_epochs,
        num_units,
        num_channels,
        timebins_per_spike
    ), np.nan)
    spike_counts = np.full((num_units, num_epochs), np.nan)


    # Temporary storage used for waveform calculations.
    _waveforms = np.empty((
        spikes_per_epoch,
        num_channels,
        timebins_per_spike
    ))

    # Iterate over epochs.
    for epoch_idx, (epoch_start, epoch_stop) in enumerate(epoch_times):

        logging.info(f"STARTING EPOCH {epoch_idx}")
        logging.info(f"\t start time : {epoch_start}")
        logging.info(f"\t stop time  : {epoch_stop}")

        # Select spikes occuring in this epoch.
        in_epoch = (
            (ks_data["spike_times"] > epoch_start) &
            (ks_data["spike_times"] < epoch_stop)
        )
        _times = ks_data["spike_times"][in_epoch]
        _clusters = ks_data["spike_clusters"][in_epoch]

        assert _times.size == _clusters.size
        logging.info(f"Epoch contains {_times.size} spikes.")

        # Iterate over cell ids.
        for cluster_idx, this_cell_id in enumerate(cell_ids):

            # Check if this cell id doesn't appear in this epoch.
            if this_cell_id not in _clusters:
                continue

            # Spike times cell id in this epoch.
            ts = _times[_clusters == this_cell_id]
            np.random.shuffle(ts)

            logging.info(f"Cluster {this_cell_id} has {ts.size} spikes.")

            # Store the total number of spikes.
            spike_counts[cluster_idx, epoch_idx] = ts.size

            # Fill waveform storage with nans (we apply nanmean and nanstd)
            # to this array to compute the summary waveforms of interest.
            _waveforms.fill(np.nan)

            # Extract waveforms.
            for i, peak_time in enumerate(ts):

                # Stop after collecting the desired number of spike waveforms.
                if i == spikes_per_epoch:
                    break

                # Find endpoints of waveform.
                start = int(peak_time - pre_samples)
                end = start + timebins_per_spike

                # Skip spikes at start or end of dataset.
                if (start < 0) or (end >= data.shape[0]):
                    continue

                # Copy data into waveforms array. 
                _waveforms[i, :, :] = data[start:end, :].T

            # Scale waveforms by gain.
            _waveforms *= metadata["uVPerBit"]

            # Compute mean and standard deviation.
            with warnings.catch_warnings():

                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_waveforms[epoch_idx, cluster_idx] = np.nanmean(_waveforms, axis=0)
                stddev_waveforms[epoch_idx, cluster_idx] = np.nanstd(_waveforms, axis=0)

                # remove offset
                mean_waveforms[epoch_idx, cluster_idx] -= (
                    mean_waveforms[epoch_idx, cluster_idx, 0, :]
                )

    return mean_waveforms, stddev_waveforms, spike_counts


def load_kilosort_data(folder, 
                       sample_rate, 
                       convert_to_seconds = True, 
                       use_master_clock = False, 
                       include_pcs = False,
                       template_zero_padding = 21):

    """
    Loads Kilosort output files from a directory

    Inputs:
    -------
    folder : String
        Location of Kilosort output directory
    sample_rate : float
        AP band sample rate in Hz
    convert_to_seconds : bool (optional)
        Flags whether to return spike times in seconds (requires sample_rate to be set)
    use_master_clock : bool (optional)
        Flags whether to load spike times that have been converted to the master clock timebase
    include_pcs : bool (optional)
        Flags whether to load spike principal components (large file)
    template_zero_padding : int (default = 21)
        Number of zeros added to the beginning of each template

    Outputs:
    --------
    spike_times : numpy.ndarray (N x 0)
        Times for N spikes
    spike_clusters : numpy.ndarray (N x 0)
        Cluster IDs for N spikes
    spike_templates : numpy.ndarray (N x 0)
        Template IDs for N spikes
    amplitudes : numpy.ndarray (N x 0)
        Amplitudes for N spikes
    unwhitened_temps : numpy.ndarray (M x samples x channels) 
        Templates for M units
    channel_map : numpy.ndarray
        Channels from original data file used for sorting
    channel_pos : numpy.ndarray (channels x 2)
        X and Z coordinates for each channel used in the sort
    cluster_ids : Python list
        Cluster IDs for M units
    cluster_quality : Python list
        Quality ratings from cluster_group.tsv file
    cluster_amplitude : Python list
        Average amplitude for each cluster from cluster_Amplitude.tsv file
    pc_features (optinal) : numpy.ndarray (N x channels x num_PCs)
        PC features for each spike
    pc_feature_ind (optional) : numpy.ndarray (M x channels)
        Channels used for PC calculation for each unit
    template_features (optional) : numpy.ndarray (N x number of features)
        projections onto template features for each spike
    """

    if use_master_clock:
        spike_times = np.load(os.path.join(folder,'spike_times_master_clock.npy'))
    else:
        spike_times = np.load(os.path.join(folder,'spike_times.npy'))
        
    spike_clusters = np.load(os.path.join(folder,'spike_clusters.npy'))
    spike_templates = np.load(os.path.join(folder, 'spike_templates.npy'))
    amplitudes = np.load(os.path.join(folder,'amplitudes.npy'))
    templates = np.load(os.path.join(folder,'templates.npy'))
    unwhitening_mat = np.load(os.path.join(folder,'whitening_mat_inv.npy'))
    channel_map = np.load(os.path.join(folder, 'channel_map.npy'))
    channel_pos = np.load(os.path.join(folder, 'channel_positions.npy'))

    if include_pcs:
        pc_features = np.load(os.path.join(folder, 'pc_features.npy'))
        pc_feature_ind = np.load(os.path.join(folder, 'pc_feature_ind.npy'))
        print("loading template_features")
        template_features = np.load(os.path.join(folder, 'template_features.npy'))

                
    templates = templates[:,template_zero_padding:,:] # remove zeros
    spike_clusters = np.squeeze(spike_clusters) # fix dimensions
    spike_times = np.squeeze(spike_times)# fix dimensions

    if convert_to_seconds and sample_rate is not None:
       spike_times = spike_times / sample_rate 
                    
    unwhitened_temps = np.zeros(templates.shape)
    
    for temp_idx in range(templates.shape[0]):
        
        unwhitened_temps[temp_idx,:,:] = np.dot(
            np.ascontiguousarray(templates[temp_idx,:,:]),
            np.ascontiguousarray(unwhitening_mat)
        )

    try:
        cluster_ids, cluster_quality = read_cluster_group_tsv(os.path.join(folder, 'cluster_group.tsv'))
    except OSError:
        cluster_ids = np.unique(spike_clusters)
        cluster_quality = ['unsorted'] * cluster_ids.size
        
    # cluster_amplitude = read_cluster_amplitude_tsv(os.path.join(folder, 'cluster_Amplitude.tsv'))


    data = {
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "spike_templates": spike_templates,
        "amplitudes": amplitudes,
        "unwhitened_temps": unwhitened_temps,
        "channel_map": channel_map,
        "channel_pos": channel_pos,
        "cluster_ids": cluster_ids,
        "cluster_quality": cluster_quality,
        # "cluster_amplitude": cluster_amplitude,
    }

    if include_pcs:
        data["pc_features"] = pc_features
        data["pc_feature_ind"] = pc_feature_ind
        data["template_features"] = template_features

    return data


def grab_files(folder, suffix):
    """
    Grabs all files in a folder ending with suffix.

    For example, if 'path/to/folder' contains a file
    'data.npy' then calling 

        grab_files("path/to/folder", ".npy")

    will return a string 'data.npy'. If there are
    multiple files it returns a list of filenames.
    """
    files = list(filter(
        lambda s: s.endswith(suffix),
        os.listdir(folder)
    ))
    return files[0] if (len(files) == 1) else files


def parse_metafile(metafile):
    """
    Parses imec metadata file.
    """

    metadata = {
        "probe_type": None,
        "sample_rate": None,
        "num_channels": None,
        "uVPerBit": (1e-6) * ((1 / 500) / pow(2, 10)) 
    }

    with open(metafile, "r") as f:
        for line in f:

            if line.startswith("imDatPrb_type"):
                p_type = int(line.split("=")[-1].strip("\n"))
                if p_type == 0:
                    metadata["probe_type"] = "NP1"
                else:
                    metadata["probe_type"] = "NP" + str(p_type)

            if line.startswith("imSampRate"):
                metadata["sample_rate"] = float(
                    line.split("=")[-1].strip("\n")
                )

            if line.startswith("nSavedChans"):
                metadata["num_channels"] = int(
                    line.split("=")[-1].strip("\n")
                )

    return metadata


def read_cluster_group_tsv(filename):

    """
    Reads a tab-separated cluster_group.tsv file from disk
    Inputs:
    -------
    filename : String
        Full path of file
    Outputs:
    --------
    IDs : list
        List of cluster IDs
    quality : list
        Quality ratings for each unit (same size as IDs)
    """

    info = np.genfromtxt(filename, dtype='str')
    cluster_ids = info[1:,0].astype('int')
    cluster_quality = info[1:,1]

    return cluster_ids, cluster_quality


def read_cluster_amplitude_tsv(filename):
    
    """
    Reads a tab-separated cluster_Amplitude.tsv file from disk
    Inputs:
    -------
    filename : String
        Full path of file
    Outputs:
    --------
    amplitudes : array
        array of average cluster amplitudes calculated by KS2
    """
    info = np.genfromtxt(filename, dtype='str')
    # don't return cluster_ids because those are already read in or 
    # derived from the spike_clusters.npy file
    # cluster_ids = info[1:,0].astype('int')
    cluster_amplitude = info[1:,1].astype('float')


    return cluster_amplitude



if __name__ == "__main__":

    ROOTPATH = "/mnt/home/awilliams/ceph/isabel"
    for jobid in range(4):
        logging.info(f"STARTING JOB: {jobid}")
        run_session(ROOTPATH, jobid)

