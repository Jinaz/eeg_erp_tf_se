#from surfer import Brain

import readData as rd 
import numpy as np 
import mne
from numpy.random import randn
from mne.datasets import fetch_fsaverage
import os.path as op
from mne.minimum_norm import make_inverse_operator, apply_inverse
from scipy import stats as stats
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)


def readin(sid='007'):
    raw, evts, eventsdict = rd.readBids(sid)
    epochsinfo = rd.get_epoched_Data(raw, evts, eventsdict)
    epochsinfo.load_data().set_eeg_reference(projection=True)

    return epochsinfo

def getinv(epochsinfo, epochslist, trans, src, bem ):
    

    fwd = mne.make_forward_solution(epochsinfo.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0)
    noise_cov = mne.compute_covariance(epochslist, tmax=0)
    #noise_cov.plot(epochs.info)

    inv_default = make_inverse_operator(epochsinfo.info, fwd, noise_cov, loose=0.2, depth=0.8)
    return inv_default


def generateComparisonTest(src, n_subjects, X):
    print('Computing adjacency.')
    adjacency = mne.spatial_src_adjacency(src)

    #    Note that X needs to be a multi-dimensional array of shape
    #    samples (subjects) x time x space, so we permute dimensions
    X2 = np.transpose(X, [2, 1, 0])

    #    Now let's actually do the clustering. This can take a long time...
    #    Here we set the threshold quite high to reduce computation.
    p_threshold = 0.001
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X2, adjacency=adjacency, n_jobs=8,
                                           threshold=t_threshold, buffer_size=None,
                                           verbose=True)
    #    Now select the clusters that are sig. at p < 0.05 (note that this value
    #    is multiple-comparisons corrected).
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    return good_cluster_inds, clu 

def plot(clu, fsave_vertices):
    print('Visualizing clusters.')

    #    Now let's build a convenient representation of each cluster, where each
    #    cluster becomes a "time point" in the SourceEstimate
    stc_all_cluster_vis = summarize_clusters_stc(clu, 
                                                 vertices=fsave_vertices,
                                                 subject='fsaverage')

    #    Let's actually plot the first "time point" in the SourceEstimate, which
    #    shows all the clusters, weighted by duration.
    # blue blobs are for condition A < condition B, red for A > B
    brain = stc_all_cluster_vis.plot(
        hemi='both', views='lateral',
        time_label='temporal extent (ms)', size=(800, 800),
        smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]))
    # brain.save_image('clusters.png')

if __name__ =="__main__":
    fsdir = fetch_fsaverage()
    subjectsdir = op.dirname(fsdir)

    subject = 'fsaverage'
    trans = 'fsaverage'
    src = op.join(fsdir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fsdir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    subjectids = rd.generateIDs()
    #subjectids = ['001','002','003','004','005','006','007','008','009','010','011','012','013']
    conditions1 = []
    conditions2 = []
    n_subj = len(subjectids)

    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    defInv = None
    
    for sid in subjectids:
        epochsinfo = readin(sid)
    

        evoked1 = epochsinfo['faces'].average()
        evoked1.resample(50, npad='auto')
        evoked2 = epochsinfo['cars'].average()
        evoked2.resample(50, npad='auto')

        default_inv = getinv(epochsinfo, epochsinfo, trans, src, bem)
        defInv = default_inv

        condition1 = apply_inverse(evoked1, default_inv, lambda2, 'MNE')
        condition2 = apply_inverse(evoked2, default_inv, lambda2, 'MNE')
        #    Let's only deal with t > 0, cropping to reduce multiple comparisons
        condition1.crop(0, None)
        condition2.crop(0, None)
        conditions1.append(condition1)
        conditions2.append(condition2)

    verts, times = conditions1[0].shape
    X = np.ones((verts, times, n_subj, 2))
    for i in range(n_subj):
        X[:,:,:,0] *= conditions1[i].data[:,:,np.newaxis]
        X[:,:,:,1] *= conditions2[i].data[:,:,np.newaxis]

    # Read the source space we are morphing to
    sroc = mne.read_source_spaces(src)
    fsave_vertices = [s['vertno'] for s in sroc]
    morph_mat = mne.compute_source_morph(
        src=defInv.get('src'), 
        spacing=fsave_vertices).morph_mat

    n_vertices_fsave = morph_mat.shape[0]
    #    We have to change the shape for the dot() to work properly
    X = X.reshape(verts, times * n_subj * 2)
    print('Morphing data.')
    X = morph_mat.dot(X)  # morph_mat is a sparse matrix
    X = X.reshape(verts, times, n_subj, 2)
    X = np.abs(X)  # only magnitude
    X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast

    del conditions1
    del conditions2
    

    good_cluster_inds, clu = generateComparisonTest(sroc, n_subj, X)
    #plot(clu, fsave_vertices)

    #    Now let's build a convenient representation of each cluster, where each
    #    cluster becomes a "time point" in the SourceEstimate
    stc_all_cluster_vis = summarize_clusters_stc(clu, 
                                                 vertices=fsave_vertices,
                                                 subject='fsaverage')
    stc_all_cluster_vis.save("stc_all")