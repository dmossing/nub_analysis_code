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

move_fn = lambda x: np.nanmax(np.abs(x[:,8:-8]),axis=-1)>1
run_fn = lambda x: np.nanmean(np.abs(x[:,8:-8]),axis=-1)>10

order1 = np.argsort((utils.nubs_active*np.array((16,1,4,8,2))[np.newaxis]).sum(1),kind='stable')
order2 = np.argsort(utils.nubs_active[order1][::-1].sum(1),kind='stable')[::-1]
evan_order_actual = order1[order2]
evan_order_apparent = np.argsort(utils.nubs_active[::-1].sum(1),kind='stable')[::-1]
nub_no = utils.nubs_active[evan_order_actual].sum(1)
parula = ListedColormap(ut.loadmat('/Users/dan/Documents/code/adesnal/matlab_parula_colormap.mat','cmap'))

def compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=utils.gen_nub_selector_v1):
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='decon',keylist=keylist,run_fn=run_fn)
    keylist = list(roi_info.keys())
    
    train_test = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        train_test[irun] = utils.select_trials(trial_info,selector,0.5)
        
    tuning = [None for irun in range(2)]
    for irun in range(2):
        tuning[irun] = utils.compute_tuning(df,trial_info,selector,include=train_test[irun])
        
    return tuning

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

def compute_bounds(dsname,keylist,datafield,run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1,pct=(2.5,97.5,50)):
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
        tuning[irun] = utils.compute_tuning_many_partitionings(df,trial_info,npartitionings,training_frac=training_frac,gen_nub_selector=selector[irun])
        
    return tuning

def compute_lkat_dfof(dsname,keylist,run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=0.4):
    tuning_dfof = compute_tuning(dsname,keylist,'F',run_fn,gen_nub_selector=gen_nub_selector)
    lkat = [None for irun in range(2)]
    for irun in range(2):
        nexpt = len(tuning_dfof[irun])
        lkat_list = [None for iexpt in range(nexpt)]
        for iexpt in range(nexpt):
            data = 0.5*(tuning_dfof[irun][iexpt][0] + tuning_dfof[irun][iexpt][1])
            lkat_list[iexpt] = np.nanmax(data-data[:,0:1],axis=1) >= dfof_cutoff
        lkat[irun] = np.concatenate(lkat_list,axis=0)
    return lkat

def compute_lkat_evan_style(dsname,keylist,run_fn=move_fn,pcutoff=0.05,dfof_cutoff=0.):
    lkat = [None for irun in range(2)]
    for irun in range(2):
        df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='F',keylist=keylist,run_fn=run_fn)
        if not irun:
            for expt in trial_info:
                trial_info[expt]['running'] = ~trial_info[expt]['running']
        roi_info = utils.test_sig_driven(df,roi_info,trial_info,pcutoff=pcutoff,dfof_cutoff=dfof_cutoff)
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

def plot_tuning_curves(targets,dsname,keylist,datafield='decon',run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=[0,1],colorbar=True,line=True,pcutoff=0.05):
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    if dfof_cutoff is None:
        lkat = [None for irun in range(2)]
    else:
        lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    for irun in run_conditions:
        plot_tuning_curves_(tuning[irun],targets[irun],lkat=lkat[irun],colorbar=colorbar,line=line)

def plot_combined_tuning_curves(targets,dsnames,keylists,datafields='decon',run_fns=None,gen_nub_selectors=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=[0,1],colorbar=True,line=True,pcutoff=0.05):
    if run_fns is None:
        run_fns = [move_fn for ifn in range(len(dsnames))]
    tuning = []
    lkat = []
    for dsname,keylist,datafield,run_fn,gen_nub_selector in zip(dsnames,keylists,datafields,run_fns,gen_nub_selectors):
        tuning.append(compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector))
        if dfof_cutoff is None:
            lkat.append(None)
        else:
            this_lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
            lkat.append(this_lkat)
    #tuning: [[irun=0,irun=1],[irun=0,irun=1]]
           
    tuning = [np.concatenate([t[irun] for t in tuning],axis=0) for irun in range(2)]
    if not lkat[0] is None:
        lkat = [np.concatenate([t[irun] for t in lkat],axis=0) for irun in range(2)]
    for irun in run_conditions[0]:
        plot_tuning_curves_(tuning[irun],targets[irun],lkat=lkat[irun],colorbar=colorbar,line=line)
def troubleshoot_lkat(dsname,keylist,datafield='decon',run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05):

    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    lkat_evan_style = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    lkat_roracle = [None for irun in range(2)]
    for irun in range(2):
        lkat_roracle[irun] = utils.compute_lkat_roracle(tuning[irun])
    return lkat_evan_style,lkat_roracle

def plot_tuning_curves_(tuning,target,lkat=None,colorbar=True,line=True):
    nexpt = len(tuning)

    lkat_roracle = utils.compute_lkat_roracle(tuning)
    if lkat is None:
        lkat = lkat_roracle
    else:
        lkat = lkat & lkat_roracle
        #print((lkat & lkat_roracle).mean())

    
    train_response = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat]
    test_response = np.concatenate([tuning[iexpt][1] for iexpt in range(nexpt)],axis=0)[lkat]
    
    ndim = len(train_response.shape)-1
    
    if ndim==1:
        train_response = train_response[:,np.newaxis,:]
        test_response = test_response[:,np.newaxis,:]
    
    fig = plt.figure(figsize=(6*ndim,6))
    ht = 6
    nsize = train_response.shape[1]
    for idim in range(nsize):
        ax = fig.add_subplot(1,nsize,idim+1)

        fig = show_evan_style(train_response[:,idim],test_response[:,idim],fig=fig,line=line,colorbar=colorbar)
    plt.tight_layout(pad=7)
    plt.savefig(target,dpi=300)

def plot_patch_no_pref(target,dsname,keylist,datafield='decon',run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,run_conditions=[0,1],colorbar=True,line=True,fig=None,cs=['C0','C1']):
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    if dfof_cutoff is None:
        lkat = [None for irun in [0,1]]
    else:
        lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
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
    
        
def show_evan_style(train_response,test_response,ht=6,cmap=parula,line=True,fig=None,colorbar=True):
    sorteach = np.argsort(train_response[:,evan_order_actual],1)[:,::-1]
    sortind = np.arange(train_response.shape[0])
    if fig is None:
        fig = plt.figure()
    for n in [0]:
        new_indexing = np.argsort(sorteach[:,n],kind='mergesort')
        sortind = sortind[new_indexing]
        sorteach = sorteach[new_indexing]
    nroi = test_response.shape[0]
    if nroi:
        img = plt.imshow(test_response[sortind][:,evan_order_actual]/test_response[sortind].max(1)[:,np.newaxis],extent=[-0.5,31.5,0,5*ht],cmap=cmap,interpolation='none')
        utils.draw_stim_ordering(evan_order_apparent,invert=True)

        show_every = 300
        plt.yticks(5*ht-np.arange(0,5*ht,show_every/nroi*5*ht),['Neuron'] + list(np.arange(show_every,nroi,show_every)))
        for this_nub_no in range(2,6):
            first_ind = np.where(nub_no==this_nub_no)[0][0]
            plt.axvline(first_ind-0.45,c='w')
            plt.axhline(5*ht*(1-np.where(sorteach[:,0]==first_ind)[0][0]/nroi),c='w',linestyle='dotted')
        plt.ylabel('Neuron')
        plt.xlabel('Stimulus')
        plt.ylim(-5,5*ht)
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
    nexpt = len(tuning)
    
    if lkat is None:
        lkat = utils.compute_lkat_roracle(tuning)
    else:
        lkat = lkat & utils.compute_lkat_roracle(tuning)

    plt.figure()
    train_response = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat]
    test_response = np.concatenate([tuning[iexpt][1] for iexpt in range(nexpt)],axis=0)[lkat]
    #test_response = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat] # temporary for testing!
    train_norm_response = train_response.copy() - train_response[:,0:1]
    train_norm_response = train_norm_response/train_norm_response.max(1)[:,np.newaxis]
    test_norm_response = test_response.copy() - test_response[:,0:1]
    test_norm_response = test_norm_response/test_norm_response.max(1)[:,np.newaxis]

    linear_pred = np.zeros_like(test_norm_response)
    single_whisker_responses = test_norm_response[:,[2**n for n in range(5)][::-1]]
    for i in range(linear_pred.shape[1]):
        linear_pred[:,i] = np.sum(single_whisker_responses*utils.nubs_active[i][np.newaxis],1)
        lin_subtracted = test_norm_response-linear_pred

    sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
    sortind = np.arange(train_norm_response.shape[0])
    for n in [3,2,1,0]:
        new_indexing = np.argsort(sorteach[:,n],kind='mergesort')
        sortind = sortind[new_indexing]
        sorteach = sorteach[new_indexing]
        
    utils.test_validity_of_linear_pred(test_norm_response,linear_pred)
    
    return lin_subtracted,sorteach,sortind

def plot_lin_subtracted_(target,lin_subtracted,sorteach,sortind):

    #fig = plt.figure(figsize=(6,6))
    #ax = fig.add_subplot(1,1,1)
    ht = 6
    ld = plt.imshow(lin_subtracted[sortind][:,evan_order_actual[6:]],extent=[-0.5,25.5,0,5*ht],cmap='bwr') #/lin_subtracted[sortind].max(1)[:,np.newaxis]
    utils.draw_stim_ordering(evan_order_apparent[6:],invert=True)

    mx = 1e-1*np.nanpercentile(np.abs(lin_subtracted),99)
    plt.ylabel('Neuron')
    plt.xlabel('Stimulus')
    plt.clim(-mx,mx)
    plt.clim(-1.05,1.05)
    plt.ylim(-5,5*ht)
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
        plt.axhline(5*ht*(1-np.where(sorteach[:,0]==first_ind)[0][0]/nroi),c='k',linestyle='dotted')
    cbaxes = plt.gcf().add_axes([0.81, 0.235, 0.03, 0.645]) 
    cb = plt.colorbar(ld,cax=cbaxes,ticks=np.linspace(-1,1,6))
    cb.ax.set_yticks(np.linspace(-1,1,6))
    cb.set_label('Normalized Linear Difference')
    plt.imshow(lin_subtracted[sortind][:,evan_order_actual[6:]],extent=[0,10,0,10],cmap='bwr')

#     plt.ylabel('neuron #')
#     plt.xlabel('stimulus #')
#     plt.clim(-1,1)
#     plt.xticks([])
#     plt.yticks([])
    plt.tight_layout(pad=7)
    
    plt.savefig(target,dpi=300)

def plot_lin_subtracted(targets,dsname,keylist=None,datafield='decon',run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,run_conditions=[0,1]):
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    if dfof_cutoff is None:
        lkat = [None for irun in range(2)]
    else:
        lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    for irun in run_conditions:
        #lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind)
        
def plot_combined_lin_subtracted(targets,dsnames,keylists,datafields='decon',run_fns=None,gen_nub_selectors=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,run_conditions=[0,1],colorbar=True,line=True):
    if run_fns is None:
        run_fns = [move_fn for ifn in range(len(dsnames))]
    tuning = []
    lkat = []
    for dsname,keylist,datafield,run_fn,gen_nub_selector in zip(dsnames,keylists,datafields,run_fns,gen_nub_selectors):
        tuning.append(compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector))
        if dfof_cutoff is None:
            lkat.append(None)
        else:
            this_lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
            lkat.append(this_lkat)
    tuning = [np.concatenate([t[irun] for t in tuning],axis=0) for irun in range(2)]
    if not lkat[0] is None:
        lkat = [np.concatenate([t[irun] for t in lkat],axis=0) for irun in range(2)]
    for irun in run_conditions[0]:
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind)
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
        lkat = (session == this_session)
        lin_subtracted,sorteach,sortind = subtract_lin(tunings[irun].loc[(session==this_session)])
        plot_sorted_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind)
        
    # reshape to iroi x ipartitioning x ipartition x istim
    # for each session
    # for each partitioning
    # compute sorting and linear difference of B based on A
    # average across partitionings
    # compute bootstrapped error bars across neurons
    
def plot_example_tuning_curves(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1):
    bounds = compute_bounds(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield)
    plot_example_tuning_curves_(bounds[irun],selected_expts,selected_rois,scale=scale)
    plt.savefig(target,dpi=300)
    
def plot_example_tuning_curves_faster(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1,pct=(16,84,50)):
    bounds = compute_bounds_faster(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield,pct=pct)
    plot_example_tuning_curves_(bounds[irun],selected_expts,selected_rois,scale=scale)
    plt.savefig(target,dpi=300)
    
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