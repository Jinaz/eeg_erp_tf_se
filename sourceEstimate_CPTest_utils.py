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

#testing code for generation with noise, not used in the end, just for checking methods
def generateX(condition1, condition2, src):
    n_vertices_sample, n_times = condition1.data.shape
    n_subjects = 2
    print('Simulating data for %d subjects.' % n_subjects)
    #    Let's make sure our results replicate, so set the seed.
    np.random.seed(0)
    X = randn(n_vertices_sample, n_times, n_subjects, 2) * 10
    X[:, :, :, 0] += condition1.data[:, :, np.newaxis]
    X[:, :, :, 1] += condition2.data[:, :, np.newaxis]

    # Read the source space we are morphing to
    sroc = mne.read_source_spaces(src)
    fsave_vertices = [s['vertno'] for s in sroc]
    morph_mat = mne.compute_source_morph(
    src=condition1, 
    spacing=fsave_vertices).morph_mat

    n_vertices_fsave = morph_mat.shape[0]
    #    We have to change the shape for the dot() to work properly
    X = X.reshape(n_vertices_sample, n_times * n_subjects * 2)
    print('Morphing data.')
    X = morph_mat.dot(X)  # morph_mat is a sparse matrix
    X = X.reshape(n_vertices_fsave, n_times, n_subjects, 2)
    X = np.abs(X)  # only magnitude
    X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast

    return X, n_subjects, sorc,fsave_vertices 


def generateComparisonTest(src, n_subjects, X):
    print('Computing adjacency.')
    adjacency = mne.spatial_src_adjacency(src)

    #    Note that X needs to be a multi-dimensional array of shape
    #    samples (subjects) x time x space, so we permute dimensions
    X = np.transpose(X, [2, 1, 0])

    #    Now let's actually do the clustering. This can take a long time...
    #    Here we set the threshold quite high to reduce computation.
    p_threshold = 0.001
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_jobs=4,
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

    subjectids = ['001','002']
    conditions1 = []
    conditions2 = []
    n_subj = 2

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    
    for sid in subjectids:
        epochsinfo = readin(sid)
    

        evoked1 = epochsinfo['faces'].average()
        evoked2 = epochsinfo['cars'].average()

        default_inv = getinv(epochsinfo, epochsinfo, trans, src, bem)

        condition1 = apply_inverse(evoked1, default_inv, lambda2, 'MNE')
        condition2 = apply_inverse(evoked2, default_inv, lambda2, 'MNE')
        conditions1.append(condition1)
        conditions2.append(condition2)

    verts, times = conditions1[0].shape
    X = np.zeros((verts, times, n_subj, 2))
    for i in range(n_subj):
        X[:,:,:,0] += conditions1[i].data[:,:,np.newaxis]
        X[:,:,:,1] += conditions2[i].data[:,:,np.newaxis]
    # Read the source space we are morphing to
    sroc = mne.read_source_spaces(src)
    fsave_vertices = [s['vertno'] for s in sroc]
    morph_mat = mne.compute_source_morph(
        src=condition1, 
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
    
    
    sroc = mne.read_source_spaces(src)
    fsave_vertices = [s['vertno'] for s in sroc]

    good_cluster_inds, clu = generateComparisonTest(sroc, n_subj, X)
    plot(clu, fsave_vertices)