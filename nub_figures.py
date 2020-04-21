#!/usr/bin/env python

import numpy as np
import pandas as pd
import h5py
import pyute as ut
from importlib import reload
reload(ut)
import pdb
import matplotlib.pyplot as plt
import nub_utils as utils
reload(utils)
import sklearn
import scipy.stats as sst
import sklearn.cluster as skc
from matplotlib.colors import ListedColormap

keylist_v1_l23 = ['session_191108_M0403','session_191119_M0293','session_200311_M0403','session_200312_M0293','session_200312_M0807']

# movement condition: max velocity during trial > 1 cm/sec
move_fn = lambda x: np.nanmax(np.abs(x[:,8:-8]),axis=-1)>1
# running condition: mean velocity during trial > 10 cm/sec
run_fn = lambda x: np.nanmean(np.abs(x[:,8:-8]),axis=-1)>10

# pupil dilated condition: pupil area > 2% of eye mask area
dilation_fn = lambda x: np.nanmedian(x[:,8:-8],axis=-1)>0.02

# reordering stimuli to convert between the geometry of V1 stimulus labels, and geometry of S1 stimulus labels
order1 = np.argsort((utils.nubs_active*np.array((16,1,4,8,2))[np.newaxis]).sum(1),kind='stable')
order2 = np.argsort(utils.nubs_active[order1][::-1].sum(1),kind='stable')[::-1]
# order of stimuli for tuning curve display purposes
evan_order_actual = order1[order2]
# order of stimuli for stimulus identity display purposes
evan_order_apparent = np.argsort(utils.nubs_active[::-1].sum(1),kind='stable')[::-1]
nub_no = utils.nubs_active[evan_order_actual].sum(1)
#parula = ListedColormap(ut.loadmat('/Users/dan/Documents/code/adesnal/matlab_parula_colormap.mat','cmap'))

# similar to parula colormap, ported to python
parula_path = '/Users/dan/Documents/code/adesnal/'
parula_filename = parula_path+'matlab_parula_colormap.mat'
parula = ListedColormap(ut.loadmat(parula_filename,'cmap'))

def compute_tuning(dsname,keylist,datafield,run_fn,dilation_fn=dilation_fn,gen_nub_selector=utils.gen_nub_selector_v1,trialwise_dfof=False):
    # extract pandas dataframe from hdf5 file
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,run_fn=run_fn,dilation_fn=dilation_fn,trialwise_dfof=trialwise_dfof)
    keylist = list(roi_info.keys())
    print(list(trial_info.keys()))
    
    # separate trials into training set and test set
    train_test = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun,dilated=1,centered=1)
        print(list(selector.keys()))
        train_test[irun] = utils.select_trials(trial_info,selector,0.5)
        
    # compute tuning curves by averaging across trials
    tuning = [None for irun in range(2)]
    for irun in range(2):
        tuning[irun] = utils.compute_tuning(df,trial_info,selector,include=train_test[irun])
        
    return tuning

###
# not used in the final analysis
def compute_bounds_faster(dsname,keylist,datafield,run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1):
    if run_fn is None:
        run_fn = move_fn
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='decon',keylist=keylist,run_fn=run_fn)
    keylist = list(roi_info.keys())
    
    bounds = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        #bounds[irun] = utils.compute_bootstrap_error(df,trial_info,selector,pct=(2.5,97.5,50))
        bounds[irun] = utils.compute_bootstrap_error_faster(df,trial_info,selector,pct=(2.5,97.5,50))
        
    return bounds

###

def compute_bounds(dsname,keylist,datafield,run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1,pct=(2.5,97.5,50)):
    # compute bootstrapped errorbars given by the percentiles in 'pct'
    # if not otherwise specified, split trials into moving and nonmoving
    if run_fn is None:
        run_fn = move_fn
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='decon',keylist=keylist,run_fn=run_fn)
    keylist = list(roi_info.keys())
    
    bounds = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        #bounds[irun] = utils.compute_bootstrap_error(df,trial_info,selector,pct=(2.5,97.5,50))
        bounds[irun] = utils.compute_bootstrap_error(df,trial_info,selector,pct=pct)
        
    return bounds

def compute_tunings(dsname,keylist,datafield,run_fn,gen_nub_selector=None,npartitionings=10,training_frac=0.5):
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='decon',keylist=keylist,run_fn=run_fn)
    keylist = list(roi_info.keys())
    
    if not gen_nub_selector is None:
        selector = [lambda x: gen_nub_selector(run=irun) for irun in range(2)]
    else:
        selector = [lambda x: utils.gen_nub_selector_v1(run=irun) for irun in range(2)]
    
    train_test = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        train_test[irun] = utils.select_trials(trial_info,selector,0.5)
        
    tuning = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        tuning[irun] = utils.compute_tuning_many_partitionings(df,trial_info,npartitionings,\
                                                               training_frac=training_frac,gen_nub_selector=selector)
        
    return tuning

def compute_lkat_dfof(dsname,keylist,run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=0.2):
    # return a boolean array identifying cells with mean dF/F >= dfof_cutoff
    tuning_dfof = compute_tuning(dsname,keylist,'F',run_fn,gen_nub_selector=gen_nub_selector)
    lkat = [None for irun in range(2)]
    for irun in range(2):
        nexpt = len(tuning_dfof[irun])
        lkat_list = [None for iexpt in range(nexpt)]
        for iexpt in range(nexpt):
            data = 0.5*(tuning_dfof[irun][iexpt][0] + tuning_dfof[irun][iexpt][1])
            lkat_list[iexpt] = np.nanmean(data-data[:,0:1],axis=1) >= dfof_cutoff
        lkat[irun] = np.concatenate(lkat_list,axis=0)
    return lkat

def compute_lkat_evan_style(dsname,keylist,run_fn=move_fn,pcutoff=0.05,dfof_cutoff=0.,datafield='decon',trialwise_dfof=False):
    # return a boolean array identifying cells significantly driven by at least one stimulus. 
    # t-test + Benjamini-Hochberg correction
    lkat = [None for irun in range(2)]
    for irun in range(2):
        df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,\
                                                             run_fn=run_fn,trialwise_dfof=trialwise_dfof)
        #if not irun:
            #for expt in trial_info:
                #trial_info[expt]['running'] = ~trial_info[expt]['running']
        roi_info = utils.test_sig_driven(df,roi_info,trial_info,pcutoff=pcutoff,dfof_cutoff=dfof_cutoff,running=irun)
        if not keylist is None:
            expts = keylist
        else:
            expts = list(roi_info.keys())
        nexpt = len(expts)
        lkat_list = [None for iexpt in range(nexpt)]
        for iexpt,expt in enumerate(expts):
            lkat_list[iexpt] = roi_info[expt]['sig_driven']
        lkat[irun] = np.concatenate(lkat_list,axis=0)
    return lkat

def plot_tuning_curves(targets,dsname,keylist,datafield='decon',run_fn=move_fn,\
                       gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=[0,1],colorbar=True,\
                       line=True,pcutoff=0.05,rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True):
    # load data and average across trials to compute tuning curves
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,trialwise_dfof=trialwise_dfof)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)
    #lkat_roracle = [utils.compute_lkat_roracle(tuning[irun]) for irun in range(2)]
    ## figure out the neurons to include in the plot
    #if dfof_cutoff is None:
    #    #lkat = [None for irun in range(2)]
    #    lkat = lkat_roracle
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    #    lkat = [lkat[irun] & lkat_roracle[irun] for irun in range(2)]
    for irun in run_conditions:
        # get final list of neurons and plot their tuning curves
        plot_tuning_curves_(tuning[irun],targets[irun],lkat=lkat[irun],colorbar=colorbar,\
                            line=line,draw_stim_ordering=draw_stim_ordering)#,rcutoff=rcutoff)

def plot_combined_tuning_curves(targets,dsnames,keylists,datafields='decon',run_fns=None,\
                                gen_nub_selectors=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=\
                                [0,1],colorbar=True,line=True,pcutoff=0.05,rcutoff=-1,trialwise_dfof=False\
                                ,draw_stim_ordering=True):
    # combine data from multiple hdf5 files and plot it
    # function to separate movement trials from non-movement trials
    if run_fns is None:
        run_fns = [move_fn for ifn in range(len(dsnames))]
    tuning = []
    lkat = []
    for dsname,keylist,datafield,run_fn,gen_nub_selector in zip(dsnames,keylists,datafields,run_fns,gen_nub_selectors):
        # iterate through combinations of cell types to plot
        this_tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                     trialwise_dfof=trialwise_dfof)
        tuning.append(this_tuning)
        this_lkat = compute_lkat_two_criteria(this_tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,\
                                              rcutoff=rcutoff,datafield=datafield,trialwise_dfof=trialwise_dfof)
        lkat.append(this_lkat)
        #if dfof_cutoff is None:
        #    lkat.append(None)
        #else:
        #    this_lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
        #    lkat.append(this_lkat)
    #tuning: [[irun=0,irun=1],[irun=0,irun=1]]
           
    tuning = [np.concatenate([t[irun] for t in tuning],axis=0) for irun in range(2)]
    if not lkat[0] is None:
        lkat = [np.concatenate([t[irun] for t in lkat],axis=0) for irun in range(2)]
    for irun in run_conditions[0]:
        plot_tuning_curves_(tuning[irun],targets[irun],lkat=lkat[irun],colorbar=colorbar,\
                            line=line,draw_stim_ordering=draw_stim_ordering)#,rcutoff=rcutoff)
def troubleshoot_lkat(dsname,keylist,datafield='decon',run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05):
    # choose neurons based on (1) significantly responding to at least one stimulus and (2) having similar tuning between training and test set, for comparison
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    lkat_evan_style = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    lkat_roracle = [None for irun in range(2)]
    for irun in range(2):
        lkat_roracle[irun] = utils.compute_lkat_roracle(tuning[irun])
    return lkat_evan_style,lkat_roracle

def plot_tuning_curves_(tuning,target,lkat=None,colorbar=True,line=True,draw_stim_ordering=True): #,rcutoff=0.5
    # apply filter based on roracle and plot
    nexpt = len(tuning)

    # keep only cells with corrcoef btw. training and test tuning curves > 0.5
    #lkat_roracle = utils.compute_lkat_roracle(tuning,rcutoff=rcutoff)
    #if lkat is None:
    #    lkat = lkat_roracle
    #else:
    #    lkat = lkat & lkat_roracle
        #print((lkat & lkat_roracle).mean())

    # keep cells based on lkat, and separate training set tuning curve from test set
    train_response = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat]
    test_response = np.concatenate([tuning[iexpt][1] for iexpt in range(nexpt)],axis=0)[lkat]
    
    ndim = len(train_response.shape)-1
    
    if ndim==1:
        train_response = train_response[:,np.newaxis,:]
        test_response = test_response[:,np.newaxis,:]
    
    nsize = train_response.shape[1]
    fig = plt.figure(figsize=(6*nsize,6))
    ht = 6
    for idim in range(nsize):
        ax = fig.add_subplot(1,nsize,idim+1)

        fig = show_evan_style(train_response[:,idim],test_response[:,idim],fig=fig,line=line,\
                              colorbar=colorbar,draw_stim_ordering=draw_stim_ordering)
    plt.tight_layout(pad=7)
    plt.savefig(target,dpi=300)

def plot_patch_no_pref(target,dsname,keylist,datafield='decon',run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,run_conditions=[0,1],colorbar=True,line=True,fig=None,cs=['C0','C1'],rcutoff=-1):
    # compute fraction of cells that prefer each number of patches, and plot them with errorbars
    # errorbars on N cells are computed as 1/sqrt(N)
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff)
    #if dfof_cutoff is None:
    #    lkat = [None for irun in [0,1]]
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    if fig is None:
        fig = plt.figure(figsize=(6,6))
    for irun in run_conditions:
        plot_patch_no_pref_(tuning[irun],lkat=lkat[irun],fig=fig,c=cs[irun])
    lbls = ['non-moving','moving']
    plt.legend([lbls[r] for r in run_conditions])
    plt.savefig(target,dpi=300)
    
def plot_patch_no_pref_(tuning,lkat=None,fig=None,c='C0'):
    nexpt = len(tuning)

    if lkat is None:
        lkat = utils.compute_lkat_roracle(tuning)

    
    train_response = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat]
    test_response = np.concatenate([tuning[iexpt][1] for iexpt in range(nexpt)],axis=0)[lkat]
    all_response = 0.5*(train_response + test_response)
    
    sorteach = np.argsort(all_response[:,evan_order_actual],1)[:,::-1]
    pref_no = nub_no[sorteach[:,0]]
    
    nos = np.arange(1,6).astype('int')
    frac_pref_no = np.zeros(nos.shape)
    err_pref_no = np.zeros(nos.shape)
    for ino,no in enumerate(nos):
        norm_by = np.sum(nub_no==no)/(31/5)/100
        frac_pref_no[ino] = np.nansum(pref_no==no)/np.sum(~np.isnan(pref_no==no))/norm_by
        err_pref_no[ino] = np.sqrt(np.nansum(pref_no==no))/np.sum(~np.isnan(pref_no==no))/norm_by
    
    if fig is None:
        fig = plt.figure(figsize=(6,6))
    fig.add_subplot(1,1,1)
    print(frac_pref_no)
    plt.errorbar(nos,frac_pref_no,err_pref_no,c=c)
    plt.scatter(nos,frac_pref_no,c=c)
    plt.xticks(nos,nos)
    plt.xlabel('Preferred number of patches')
    plt.ylabel('% Neurons (norm. by # of stim)')
    #plt.tight_layout(pad=7)
    
        
def show_evan_style(train_response,test_response,ht=6,cmap=parula,line=True,fig=None,colorbar=True,draw_stim_ordering=True):
    # show stimuli in order of increasing patch no (columns) and ROIs in order of preferred stim (rows)
    # for each ROI (rows), stimulus indices in order of decreasing preference (columns are preference rank)
    sorteach = np.argsort(train_response[:,evan_order_actual],1)[:,::-1]
    # sort ROIs according to their #1 ranked stimulus in the training set
    sortind = np.arange(train_response.shape[0])
    if fig is None:
        fig = plt.figure()
    for n in [3,2,1,0]:
        new_indexing = np.argsort(sorteach[:,n],kind='mergesort')
        sortind = sortind[new_indexing]
        sorteach = sorteach[new_indexing]
    nroi = test_response.shape[0]
    if nroi:
        # plot test set tuning curves normalized to their max index in the test set
        img = plt.imshow(test_response[sortind][:,evan_order_actual]/np.abs(test_response[sortind].max(1)[:,np.newaxis]),\
                         extent=[-0.5,31.5,0,5*ht],cmap=cmap,interpolation='none',vmin=0,vmax=1)
        if draw_stim_ordering:
            # underneath, plot graphic of stimuli in the same order as the columns
            utils.draw_stim_ordering(evan_order_apparent,invert=True)

        show_every = 300
        plt.yticks(5*ht-np.arange(0,5*ht,show_every/nroi*5*ht),['Neuron'] + list(np.arange(show_every,nroi,show_every)))
        for this_nub_no in range(2,6):
            first_ind = np.where(nub_no==this_nub_no)[0][0]
            plt.axvline(first_ind-0.45,c='w')
            plt.axhline(5*ht*(1-np.where(sorteach[:,0]==first_ind)[0][0]/nroi),c='w',linestyle='dotted')
        plt.ylabel('Neuron')
        plt.xlabel('Stimulus')
        if draw_stim_ordering:
            plt.ylim(-5,5*ht)
        else:
            plt.ylim(0,5*ht)
        plt.xlim(0.5,31.5)
        plt.xticks([])

        plt.text(-5,5*ht+1,'# patches: ')
        lbl_locs = [2.75,10.25,20.25,27.75,30.75]
        for inub,this_nub_no in enumerate(range(1,6)):
            plt.text(lbl_locs[inub],5*ht+1,this_nub_no)
        if line:
            plt.plot((0.5,31.5),(5*ht,0),c='k')
        if colorbar:
            cbaxes = fig.add_axes([0.12, 0.28, 0.03, 0.52])
            cb = plt.colorbar(img,cax=cbaxes)
            cbaxes.yaxis.set_ticks_position('left')
            cb.set_label('Normalized Response')
            cbaxes.yaxis.set_label_position('left')
        plt.tight_layout(pad=7)
    return fig
     
def subtract_lin(tuning,lkat=None):
    # compute evoked response, norm to max evoked response, subtract linear sum of single-patch normed responses
    nexpt = len(tuning)
    
    # to plot: cells that either have corrcoef training vs. test > 0.5, or both that and some other criterion
    #if lkat is None:
    #    lkat = utils.compute_lkat_roracle(tuning)
    #else:
    #    lkat = lkat & utils.compute_lkat_roracle(tuning)

    #plt.figure()
    train_response = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat]
    test_response = np.concatenate([tuning[iexpt][1] for iexpt in range(nexpt)],axis=0)[lkat]
    ndim = len(train_response.shape)
    if ndim==2:
        train_response = train_response[:,np.newaxis,:]
        test_response = test_response[:,np.newaxis,:]
        ndim = 3
    # convert to evoked event rates, normalized to max
    fudge = 1e-4
    train_norm_response = train_response.copy() - train_response[:,:,0:1]
    train_norm_response = train_norm_response/(fudge + train_norm_response[:,:,1:].max(2)[:,:,np.newaxis])
    test_norm_response = test_response.copy() - test_response[:,:,0:1]
    test_norm_response = test_norm_response/(fudge + test_norm_response[:,:,1:].max(2)[:,:,np.newaxis])

    # 
    # linear_pred: (nroi,[nsize,]nstim)
    linear_pred = np.zeros_like(test_norm_response)
    slicer = [slice(None) for idim in range(ndim)]
    slicer[-1] = [2**n for n in range(5)][::-1]
    # single_whisker_responses: (nroi,[nsize,]nsingle)
    single_whisker_responses = test_norm_response[slicer]
    #single_whisker_responses = test_norm_response[:,[2**n for n in range(5)][::-1]]
    slicer_output = [slice(None) for idim in range(ndim)]
    slicer_input = [np.newaxis for idim in range(ndim-1)]
    slicer_input = slicer_input+[slice(None)]
    # utils.nubs_active: (nstim,nsingle)
    for istim in range(linear_pred.shape[-1]): # nstim
        slicer_output[-1] = istim
        linear_pred[slicer_output] = np.sum(single_whisker_responses*utils.nubs_active[istim][slicer_input],-1)
        lin_subtracted = test_norm_response-linear_pred

    # sorteach: ith row gives indices to sort responses of ith neuron in descending order
    slicer[-1] = evan_order_actual
    slicer1 = [slice(None) for idim in range(ndim)]
    slicer1[-1] = slice(None,None,-1)
    #sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
    sorteach = np.argsort(train_norm_response[slicer],-1)[slicer1]
    # sortind: array that sorts neurons according to preferred stimulus
    if ndim==2:
        sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
        sortind = np.arange(train_norm_response.shape[0])
        for n in [3,2,1,0]: #[3,2,1,0]:
            slicer[-1] = 0
            new_indexing = np.argsort(sorteach[slicer],kind='mergesort')
            sortind = sortind[new_indexing]
            sorteach = sorteach[new_indexing]
            
        # confirm that linear prediction is equal to the sum of single neuron 
        # normed evoked responses
        utils.test_validity_of_linear_pred(test_norm_response,linear_pred)
    if ndim==3:
        nsize = linear_pred.shape[1]
        sorteach = np.array([np.argsort(train_norm_response[:,isize,evan_order_actual],1)[:,::-1]\
                             for isize in range(nsize)])
        sortind = np.array([np.arange(train_norm_response.shape[0]) for isize in range(nsize)])
        for isize in range(nsize):
            for n in [0]: #[3,2,1,0]:
                new_indexing = np.argsort(sorteach[isize][:,n],kind='mergesort')
                sortind[isize] = sortind[isize][new_indexing]
                sorteach[isize] = sorteach[isize][new_indexing]
                
            # confirm that linear prediction is equal to the sum of single neuron 
            # normed evoked responses
            utils.test_validity_of_linear_pred(test_norm_response[:,isize],linear_pred[:,isize])
        sortind = sortind.T
        print(sortind.shape)
        sorteach = sorteach.transpose((1,0,2))
        print(sorteach.shape)
    
    return lin_subtracted,sorteach,sortind

def plot_lin_subtracted_(target=None,lin_subtracted=None,sorteach=None,sortind=None,draw_stim_ordering=True):
    # plot heat map of linear differences, with stim visualization below, aligned to columns
    #fig = plt.figure(figsize=(6,6))
    #ax = fig.add_subplot(1,1,1)
    ndim = len(lin_subtracted.shape)-1
    if ndim==1:
        lin_subtracted = lin_subtracted[:,np.newaxis,:]
        sorteach = sorteach[:,np.newaxis,:]
        sortind = sortind[:,np.newaxis]
    nsize = lin_subtracted.shape[1]
    plt.figure(figsize=(6*nsize,6))
    for isize in range(nsize):
        plt.subplot(1,nsize,isize+1)
        ht = 6
        ld = plt.imshow(lin_subtracted[sortind[:,isize]][:,isize,evan_order_actual[6:]],extent=[-0.5,25.5,0,5*ht],cmap='bwr') #/lin_subtracted[sortind].max(1)[:,np.newaxis]
        if draw_stim_ordering:
            utils.draw_stim_ordering(evan_order_apparent[6:],invert=True)
    
        #mx = 1e-1*np.nanpercentile(np.abs(lin_subtracted),99)
        plt.ylabel('Neuron')
        plt.xlabel('Stimulus')
        #plt.clim(-mx,mx)
        plt.clim(-1.05,1.05)
        if draw_stim_ordering:
            plt.ylim(-5,5*ht)
        else:
            plt.ylim(0,5*ht)
        plt.xlim(-0.5,25.5)
        plt.xticks([])
        plt.yticks([])
        nroi = lin_subtracted.shape[0]
        show_every = 300
        plt.yticks(5*ht-np.arange(0,5*ht,show_every/nroi*5*ht),['Neuron'] + list(np.arange(show_every,nroi,show_every)))
        plt.yticks([])
        for this_nub_no in range(2,6):
            first_ind = np.where(nub_no[6:]==this_nub_no)[0][0]
            plt.axvline(first_ind-0.5,c='k')
        for this_nub_no in range(2,6):
            first_ind = np.where(nub_no==this_nub_no)[0][0]
            plt.axhline(5*ht*(1-np.where(sorteach[:,isize,0]==first_ind)[0][0]/nroi),c='k',linestyle='dotted')
        #cbaxes = plt.gcf().add_axes([0.81, 0.235, 0.03, 0.645]) 
        #cb = plt.colorbar(ld,cax=cbaxes,ticks=np.linspace(-1,1,6))
        #cb.ax.set_yticks(np.linspace(-1,1,6))
        #cb.set_label('Normalized Linear Difference')
        #plt.imshow(lin_subtracted[sortind[:,isize]][:,isize,evan_order_actual[6:]],extent=[0,10,0,10],cmap='bwr')

#     plt.ylabel('neuron #')
#     plt.xlabel('stimulus #')
#     plt.clim(-1,1)
#     plt.xticks([])
#     plt.yticks([])
    plt.tight_layout(pad=7)
    if not target is None:
        plt.savefig(target,dpi=300)

def compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=None,pcutoff=0.05,rcutoff=-1,\
                              datafield='decon',trialwise_dfof=False):
    # return boolean arrays of cells that meet both (1) a criterion on correlation coefficient between 
    # training and test set, and (2) a criterion on significance of responses
    lkat_roracle = [utils.compute_lkat_roracle(tuning[irun],rcutoff=rcutoff) for irun in range(2)]
    print([lkat_roracle[irun].mean() for irun in range(2)])
    # figure out the neurons to include in the plot
    if dfof_cutoff is None:
        #lkat = [None for irun in range(2)]
        lkat = lkat_roracle
    else:
        lkat_sig_driven = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,datafield=datafield,trialwise_dfof=trialwise_dfof)
        print([lkat_sig_driven[irun].mean() for irun in range(2)])
        lkat = [lkat_sig_driven[irun] & lkat_roracle[irun] for irun in range(2)]
    return lkat
    

def plot_lin_subtracted(targets,dsname,keylist=None,datafield='decon',run_fn=move_fn,\
                        gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                        run_conditions=[0,1],rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True):
    # compute and plot linear difference
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,trialwise_dfof=trialwise_dfof)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)
    #if dfof_cutoff is None:
    #    lkat = [None for irun in range(2)]
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    for irun in run_conditions:
        #lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind,draw_stim_ordering=draw_stim_ordering)
    return lin_subtracted,sorteach,sortind
        
def plot_combined_lin_subtracted(targets,dsnames,keylists,datafields='decon',run_fns=None,\
                                 gen_nub_selectors=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                                 run_conditions=[0,1],colorbar=True,line=True,\
                                 rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True):
    # combine two datasets, and compute and plot linear differences
    if run_fns is None:
        run_fns = [move_fn for ifn in range(len(dsnames))]
    tuning = []
    lkat = []
    for dsname,keylist,datafield,run_fn,gen_nub_selector in zip(dsnames,keylists,datafields,run_fns,gen_nub_selectors):
        this_tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                     trialwise_dfof=trialwise_dfof)
        tuning.append(this_tuning)
        this_lkat = compute_lkat_two_criteria(this_tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,\
                                              rcutoff=rcutoff,datafield=datafield,trialwise_dfof=trialwise_dfof)
        lkat.append(this_lkat)
        #if dfof_cutoff is None:
        #    lkat.append(None)
        #else:
        #    this_lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    tuning = [np.concatenate([t[irun] for t in tuning],axis=0) for irun in range(2)]
    if not lkat[0] is None:
        lkat = [np.concatenate([t[irun] for t in lkat],axis=0) for irun in range(2)]
    for irun in run_conditions[0]:
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind,draw_stim_ordering=draw_stim_ordering)
        #plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach[lkat[irun]],sortind[lkat[irun]])
    return lkat

    #     plt.subplot(1,2,2)
#         lin_subtracted,_,_ = subtract_lin(test_response)
#         plt.imshow(lin_subtracted[sortind][:,evan_order_actual[6:]],extent=[0,10,0,10],cmap='bwr') #/lin_subtracted[sortind].max(1)[:,np.newaxis]

#         plt.ylabel('neuron #')
#         plt.xlabel('stimulus #')
#         plt.clim(-1,1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.tight_layout()

### 
# not used in final analysis

def sort_lin_subtracted(lin_subtracted,sorteach,sortind):
    lin_subtracted_sort = lin_subtracted.copy()
    for iroi in range(lin_subtracted.shape[0]):
        lin_subtracted_sort[iroi] = lin_subtracted[sortind][iroi][evan_order_actual][sorteach[iroi]]
    return lin_subtracted_sort

def plot_sorted_lin_subtracted_(target,lin_subtracted,sorteach,sortind):
    #if fig is None:
    #    fig = plt.figure()
    lin_subtracted_sort = sort_lin_subtracted(lin_subtracted,sorteach,sortind)
    lb,ub,mn = ut.bootstrap(lin_subtracted_sort[nub_no[sorteach[:,0]]>1],axis=0,pct=(16,84,50),fn=np.nanmean)
    plt.fill_between(np.arange(1,33),lb,ub)
    plt.xticks((1,5,10,15,20,25,30))
    plt.xlabel('Stimulus rank')
    plt.ylabel('Normalized linear difference')
    plt.axhline(0,linestyle='dotted',c='k')
    plt.savefig(target,dpi=300)
    
def plot_sorted_lin_subtracted(targets,dsname,keylist=None,datafield='decon',run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1):
    tunings = compute_tunings(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    for irun in range(2):
        npartitioning = len(tunings[irun].partitioning.unique())
        sessions = tunings[irun].session_id.unique()
        lin_subtracted = [None for ipartitioning in range(npartitioning)]
        for ipartitioning in range(npartitioning):
            this_tuning = [None for session in sessions]
            for isession,session in enumerate(sessions):
                this_tuning[isession] = [None for ipartition in range(2)]
                for ipartition in range(2):
                    this_tuning[isession][ipartition] = tunings[irun].loc[(tunings[irun].session_id==session) \
                                                 & (tunings[irun].partitioning==ipartitioning) \
                                                 & (tunings[irun].partition==ipartition)].iloc[:,:32].to_numpy()
            this_lin_subtracted,this_sorteach,this_sortind = subtract_lin(this_tuning)
            this_lin_subtracted,this_sorteach,this_sortind = this_lin_subtracted[0],this_sorteach[0],this_sortind[0]
            lin_subtracted[ipartitioning] = np.zeros(this_lin_subtracted.shape)
            for iroi in range(this_lin_subtracted.shape[0]):
                lin_subtracted[ipartitioning][iroi] = this_lin_subtracted[iroi][this_sorteach[iroi]]
            #lin_subtracted[ipartitioning] = this_lin_subtracted[this_sorteach]
        #lin_subtracted = np.concatenate(lin_subtracted,axis=0)
        #sorteach = np.array([np.arange(32) for iroi in range(lin_subtracted.shape[0])])
        #sortind = np.arange(lin_subtracted.shape[0])
        print(lin_subtracted.shape)
        print(np.nanmax(lin_subtracted))
        lin_subtracted = np.nanmean(np.array(lin_subtracted),axis=0)
        sorteach = np.array([np.arange(32) for iroi in range(lin_subtracted.shape[0])])
        sortind = np.arange(lin_subtracted.shape[0])
        plot_sorted_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind)
    # reshape to iroi x ipartitioning x ipartition x istim
    # for each session
    # for each partitioning
    # compute sorting and linear difference of B based on A
    # average across partitionings
    # compute bootstrapped error bars across neurons
    
###
    
def plot_example_tuning_curves(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1):
    bounds = compute_bounds(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield)
    plot_example_tuning_curves_(bounds[irun],selected_expts,selected_rois,scale=scale)
    plt.savefig(target,dpi=300)
    
###
# not used in final analysis
def plot_example_tuning_curves_faster(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1,pct=(16,84,50)):
    bounds = compute_bounds_faster(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield,pct=pct)
    plot_example_tuning_curves_(bounds[irun],selected_expts,selected_rois,scale=scale)
    plt.savefig(target,dpi=300)
    
###
    
def plot_example_tuning_curves_(bounds,selected_expts,selected_rois,scale=1,ylim=(0,0.5),xlim=None,to_plot=slice(None)):
    iiroi = 0
    nroi = len(selected_rois)
    plt.figure(figsize=(10*scale,len(selected_expts)*scale))
    these_numbers_of_patches = [np.arange(1,6), np.arange(6,16), np.arange(16,26), np.arange(26,31), np.array((31,))]
    for iexpt,iroi in zip(selected_expts,selected_rois):#
        plt.subplot(nroi,1,iiroi+1)
        yerr_down = bounds[iexpt][2][iroi][evan_order_actual]-bounds[iexpt][0][iroi][evan_order_actual]
        yerr_up = bounds[iexpt][1][iroi][evan_order_actual]-bounds[iexpt][2][iroi][evan_order_actual]
        yerr = np.concatenate((yerr_down[np.newaxis],yerr_up[np.newaxis]),axis=0)
        for itn,tn in enumerate(these_numbers_of_patches):
        # plt.fill_between(np.arange(32),bounds[0][0][0][iroi][evan_order_actual],bounds[0][0][1][iroi][evan_order_actual])
            plt.errorbar(tn,bounds[iexpt][2][iroi][evan_order_actual][tn],c=parula(itn/5),yerr=yerr[:,tn],capsize=2,fmt='.')
        plt.plot(0.5+np.arange(32),np.zeros((32,)),linestyle='dashed',c='k')
        plt.axis('off')
        if not ylim is None:
            plt.ylim(ylim)
        if not xlim is None:
            plt.xlim(xlim)
        iiroi = iiroi+1
    
def plot_example_lin_subtracted(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1,bounds=None,run_all=True,pct=(16,84,50)):
    if bounds is None:
        bounds = compute_bounds(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield,pct=pct)
    if run_all:
        lin_subtracted_bounds = utils.compute_lin_subtracted_bounds(bounds[irun])
        plot_example_tuning_curves_(lin_subtracted_bounds,selected_expts,selected_rois,scale=scale,ylim=(-1.4,1.5),xlim=(5.5,32))#,to_plot=slice(6,None))
        plt.savefig(target,dpi=300)
        return
    return bounds