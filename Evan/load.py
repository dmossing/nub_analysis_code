#!/usr/bin/env python

import os, errno
import numpy as np
import h5py
from scipy.io import loadmat


def get_files(index=0, string="_catch", directory="../Data"):
    '''
    Filenames for single dataset. 
    
    Parameters
    ----------
    index : int (0-12)
        Index of dataset to load.
    string: str, optional 
        Appended string added to basename when searching for ROI files.
    directory: str, optional
        File path to prefix to basename.
    '''
    files = []
    files.append("7142_220_002")  # 0
    files.append("6994_210_000")  # 1
    files.append("7120_250_003")  # 2
    files.append("7197_160_001")  # 3 - bad
    files.append("7734_338_000")  # 4
    files.append("7734_308_001")  # 5
    files.append("7736_300_000")  # 6
    files.append("7736_265_001")  # 7
    files.append("7737_291_000")  # 8
    files.append("7737_326_001")  # 9
    files.append("9445_180_005")  # 10
    files.append("9019_165_000")  # 11
    files.append("9025_180_002")  # 12

    filebase = os.path.join(directory, files[int(index)])
    exp_file = filebase + ".exp"

    daq_file = filebase + ".bin"

    config_file = filebase + ".mat"

    if os.path.isfile(filebase + string + ".rois"):
        roi_files = [filebase + string + ".rois"]
    elif os.path.isfile(filebase + "_depth1" + string + ".rois"):
        roi_files = [
            filebase + "_depth%01d" % d + string + ".rois" for d in [1, 2, 3, 4]
        ]
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filebase + string
        )

    return {"exp": exp_file, "daq": daq_file, "cfg": config_file, "roi": roi_files}


def get_dataset(name='S1 L2/3'):
    '''
    Filenames of all data for a given dataset. 
    
    Parameters
    ----------
    name : {'S1 L2/3', 'S1 L4', 'S1 L2/3 anes'}
        Dataset identifier.
    '''
    if name == 'S1 L2/3':
        indices = range(0,3)
    elif name == 'S1 L4':
        indices = range(4,10)
    elif name == 'S1 L2/3 anes':
        indices = range(10,13)
    files = []
    for index in indices:
        files.append(get_files(index))
    files = {k: [f[k] for f in files] for k in files[0]} # unpack list of dicts to single dict of lists
    files['num'] = len(indices)
    #files['rid'] = np.repeat(np.arange(len(files['roi'])), [len(l) for l in files['roi']]) # index of ROI file into dataset
    #files['roi'] = [f for l in files['roi'] for f in l] # unpack list ROI file lists
    return(files)


def load_experiment(filename):
    '''
    Load experiment information. 
    
    Parameters
    ----------
    filename : str
        .exp file to load from.
    '''
    with h5py.File(filename, "r") as h5f:
        stim_id = h5f["/TrialInfo/StimID"][:].astype("int")
        stim_dict = h5f["/Experiment/stim/stim"][:].transpose().astype("bool")
        trial_index = h5f["TrialIndex"][:].astype("int") - 1  # convert to 0-based indexing
    return stim_dict, np.squeeze(stim_id), np.squeeze(trial_index)


def load_config(filename):
    '''
    Load configuration information. 
    
    Parameters
    ----------
    filename : str
        Scanbox generated .mat file to load from.
    '''
    info = loadmat(filename)
    trigger_frame = info["info"]["frame"][0][0]
    # trigger_line = info["info"]["line"][0][0]
    trigger_ID = info["info"]["event_id"][0][0]
    trigger_frame = trigger_frame[trigger_ID == 1]
    try:
        num_depths = len(info["info"]["otwave"][0][0][0])
    except:
        num_depths = 1
    frame_rate = 15.46/num_depths
    return trigger_frame, num_depths, frame_rate


def load_roi_data(filenames: list, location: str):
    '''
    Load ROI data. 
    
    Parameters
    ----------
    filenames : list
        List of .roi files to load from.
    location: str
        Path to data in hdf5 file (appended to 'ROIdata/rois/').
    '''
    data = []
    for f in filenames:
        h5f = h5py.File(f, "r")
        n = len(h5f["ROIdata/rois/" + location])
        dim = np.shape(
            h5f[h5f["ROIdata/rois/" + location][0][0]][:]
        )  # assumes all data has same dimensions
        data.append(np.zeros([n] + list(dim)))
        for n in np.arange(n):
            data[-1][n,] = h5f[h5f["ROIdata/rois/" + location][n][0]][:]
        h5f.close()
    data = np.concatenate(data[:], axis=0).squeeze()
    return data


def load_roi_traces(filenames):
    fluorescence = load_roi_data(filenames, location="rawdata")
    neuropil = load_roi_data(filenames, location="rawneuropil")
    return fluorescence, neuropil


def load_roi_trial_means(filenames):
    return load_roi_data(filenames, location="stimMean")


def load_roi_tuning_curves(filenames):
    curves = load_roi_data(filenames, location="curve")
    ci = load_roi_data(filenames, location="CI95")
    ci = np.abs(
        ci - curves[:, :, np.newaxis]
    )  # make bootstrapped values relative distance from avg
    return curves, ci


def load_roi_centroids(filenames):
    return load_roi_data(filenames, location="centroid")


def load_num_depths(filename):
    '''
    Load number of depths acquired. 
    
    Parameters
    ----------
    filename : str
        .roi file to load from.
    '''
    with h5py.File(filename, "r") as h5f:
        return h5f["ROIdata/Config/Depth"][0][0].astype(int)


def depth_indices(num_frames: int, num_depths=4, depth=0):
    '''
    Load frame indices captured at a given depth. 
    
    Parameters
    ----------
    num_frames : int
        Number of frames captured in the experiment/file.
    num_depths : int
        Number of depths captured in the experiment/file.
    depth : int (0 <= depth < num_depths)
        Index of depth to return frame indices for.
    '''
    return np.arange(depth, num_frames, num_depths)
