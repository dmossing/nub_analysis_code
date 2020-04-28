#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def compute_neuropil_coeff(fluorescence, neuropil):
    coeff = np.min(fluorescence / neuropil, axis=1)
    coeff[coeff > 1] = 1
    return coeff


def correct_neuropil(fluorescence, neuropil, neuropil_coeff):
    if neuropil_coeff.ndim==1:
        neuropil_coeff = neuropil_coeff[:, np.newaxis]
    return fluorescence - neuropil_coeff * neuropil


def baseline_fluorescence(traces, stim_id, trigger_frame, num_frames=4, depth_id=None):
    assert(len(stim_id)==len(trigger_frame)/2)
    trigger_frame = np.reshape(trigger_frame, (-1,2))
    
    trials = np.argwhere(stim_id==0)
    windows = np.empty((len(trials), 2), dtype=int)
    for idx, t in enumerate(trials):
        if t != len(stim_id)-1:
            windows[idx,1] = trigger_frame[t+1,0]
        else:
            windows[idx,0] = trigger_frame[t,1] # end of 'stim' period
            windows[idx,1] = windows[idx,0] + min(np.median(np.diff(windows[:-1,:], 1)), np.size(traces,1))
        windows[idx,0] = windows[idx,1] - num_frames
        if depth_id is not None:
            windows[idx,0] = np.argmax(depth_id>=windows[idx,0])
            windows[idx,1] = np.where(depth_id<=windows[idx,1])[0][-1]
    vals = [np.nanmean(traces[:,windows[t,0]:windows[t,1]], axis=1) for t in range(windows.shape[0])] # mean over time
    vals = np.stack(vals)
    baseline = np.nanmean(vals, axis=0) # mean over trials
    baseline = baseline[:,np.newaxis]
    return baseline


def dfof(f, f_0):
    if f_0.ndim==1:
        f_0 = f_0[:,np.newaxis]
    return((f-f_0)/f_0)


def trial_align(traces, stim_start, depth_id, n_before=16, n_after=31):
    trial_aligned = []
    for trial, frame in enumerate(stim_start):
        index = np.where(depth_id >= frame)[0][0]
        trial_aligned.append(traces[:, index - n_before : index + n_after])
    trial_aligned = np.stack(trial_aligned[:], axis=2)
    return trial_aligned


def time_avg(trial_aligned, frame_index=np.arange(17, 32)):
    return np.mean(trial_aligned[:, frame_index, :]).squeeze()


def compute_tuning_curve(trial_means, stim_id):
    '''
    Compute tuning curves by taking mean over trials. 
    
    Parameters
    ----------
    trial_means : ndarray (np.shape = (ROIs, trials))
        Measured trial responses.
    stim_id : vector (len = trials)
        ID of stimulus presented for that trial.
    '''
    ids = np.unique(stim_id)
    curves = np.empty((np.size(trial_means,0), len(ids)))
    ci     = np.empty((np.size(trial_means,0), len(ids)))
    for idx, i in enumerate(ids):
        curves[:,idx] = np.nanmean(trial_means[:,stim_id==i], axis=1)
        ci[:,idx] = 1.96 * np.nanstd(trial_means[:,stim_id==i], axis=1) / (np.sqrt(sum(stim_id==i)))
    return curves, ci


def selectivity(curves):
    '''
    Compute vector selectivity. 
    
    Parameters
    ----------
    curves : ndarray (np.shape = (ROIs, stimuli))
        Calculated tuning curves.
    '''
    sel = lambda x: 1 - (np.sqrt(sum(abs(x)**2))/max(x)-1)/(np.sqrt(len(x))-1)
    if np.ndim(curves)==1:
        return sel(curves)
    else:
        return np.apply_along_axis(sel, axis=1, arr=curves)

    
def compute_linear_difference(trial_means, stim_id, stim_dict, k=10000, alpha=.05, random_state=42):
    '''
    Compute difference from linear summation. 
    
    Parameters
    ----------
    trial_means : ndarray (np.shape = (ROIs, trials))
        Measured trial responses.
    stim_id : vector (len = trials)
        ID of stimulus presented for that trial.
    stim_dict : ndarray (np.shape = (stimuli, pistons))
        Mapping of which piston is present in which stimulus.
    k : int
        Number of bootstrap repititions.
    alpha : float (0-1)
        Confidence interval cutoff.
    '''
    n_rois, n_trials = np.shape(trial_means)
    
    # Generate distributions
    ids = np.unique(stim_id)
    curves = np.empty((n_rois, k+1, len(ids)))
    random_state = np.random.RandomState(seed=random_state)
    for i, s in enumerate(ids):
        ind = np.nonzero(stim_id==s)[0]
        for fold in range(k+1):
            if fold: inds = resample(ind, random_state=random_state)
            else   : inds = ind
            curves[:,fold,i] = np.nanmean(trial_means[:,inds], axis=1)
    
    # Compute linear difference
    ids = np.nonzero(np.sum(stim_dict,1)>1)[0]
    actual = np.empty((n_rois, k+1, len(ids)))
    summation = np.empty((n_rois, k+1, len(ids)))
    if k: ci = np.empty((n_rois, len(ids), 2))
    else: ci = None
    for i, s in enumerate(ids):
        actual[:,:,i] = curves[:,:,s]
        components = np.nonzero(stim_dict[s,:])[0]
        summation[:,:,i] = np.nansum(curves[:,:,components], axis=-1)
        if k:
            diff = actual[:,1:,i] - summation[:,1:,i]
            ci[:,i,:] = np.percentile(diff, [alpha/2*100, 100-alpha/2*100], axis=1).T
    linear_difference = actual[:,0,:] - summation[:,0,:]
    if k: ci = ci - linear_difference[:,:,np.newaxis] # make CI diff from mean rather than actual values
    
    return linear_difference, ci, actual[:,0,:], summation[:,0,:]


def sort_tuning_curves(curves):
    best_stim = np.argmax(curves, axis=1)
    # best_resp = np.amax(curves, axis=1)
    sel = selectivity(curves)
    sort_index = np.lexsort((1-sel, best_stim))
    return sort_index
    
    
def cv_sort(trial_means, stim_id, stim_dict, test_size=.5, random_state=42, k=0, alpha=.05):
    '''
    Sort tuning curves via cross-validation. 
    
    Parameters
    ----------
    trial_means : list of ndarrays (np.shape = (ROIs, trials))
        Measured trial responses.
    stim_id : list of vectors (len = trials)
        ID of stimulus presented for that trial.
    stim_dict : list of ndarrays (np.shape = (stimuli, pistons))
        Mapping of which piston is present in which stimulus.
    '''
    curves_train, curves_test, ld_test = [], [], []
    for idx, (means, stim, d) in enumerate(zip(trial_means, stim_id, stim_dict)):
        train, test = train_test_split(np.arange(len(stim)), test_size=test_size, random_state=random_state, shuffle=True, stratify=stim)
        curves_train.append(compute_tuning_curve(means[:,train], stim[train]))
        curves_test.append( compute_tuning_curve(means[:,test] , stim[test] ))
        ld_test.append(compute_linear_difference(means[:,test] , stim[test], d, k=k, alpha=alpha))
    
    curves_train = np.concatenate([f[0] for f in curves_train], axis=0)
    curves    = np.concatenate([f[0] for f in curves_test], axis=0)
    curves_ci = np.concatenate([f[1] for f in curves_test], axis=0)
    ld        = np.concatenate([f[0] for f in ld_test], axis=0)
    if k: ld_ci = np.concatenate([f[1] for f in ld_test], axis=0)
    else: ld_ci = None
    
    sort_index = sort_tuning_curves(curves_train)
    curves    = curves[sort_index,:]
    curves_ci = curves_ci[sort_index,:]
    ld        = ld[sort_index,:]
    if k: ld_ci = ld_ci[sort_index,:,:]
    
    return curves, curves_ci, ld, ld_ci


