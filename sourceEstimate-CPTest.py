import os.path as op
import mne
from mne.datasets import fetch_fsaverage
import readData as rd
from matplotlib import pyplot as plt 
from surfer import Brain
from mne.minimum_norm import make_inverse_operator, apply_inverse
import source_Estimate as ses 

if __name__ == "__main__":
    fsdir = fetch_fsaverage()
    subjectsdir = op.dirname(fsdir)

    subject = 'fsaverage'
    trans = 'fsaverage'
    src = op.join(fsdir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fsdir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    epochsinfo = None
    faces = []
    evfa = []
    cars = []
    evca = []

    subjectids = rd.generateIDs()
    newids = ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020']
    for sid in newids:

        raw, evts, evtsdict = rd.readBids(sid)
        epochs = rd.get_epoched_Data(raw, evts, evtsdict)
        epochs_faces, epochs_cars, sc_faces, sc_cars = ses.get_epochs(epochs)
        #for combined

        epochs_cars.set_eeg_reference(projection=True)
        epochs_faces.set_eeg_reference(projection=True)

        evfaces = epochs_faces.average()
        evcars = epochs_cars.average()

        faces.append(epochs_faces)
        evfa.append(evfaces)

        cars.append(epochs_cars)
        evca.append(evcars)

        epochsinfo = epochs.copy()


    gafa = mne.grand_average(evfa)
    gaca = mne.grand_average(evca)
    del evfa 
    del evca 
    
    comparison = mne.combine_evoked([gafa, gaca], [1,-1])

    inittime = comparison.copy().crop(timestart,timeend).pick(['PO8']).get_peak()[1]

    fwd = mne.make_forward_solution(epochsinfo, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0)
    noise_cov = mne.compute_covariance(epochslist, tmax=0)
    #noise_cov.plot(epochs.info)

    inv_default = make_inverse_operator(epochsinfo, fwd, noise_cov, loose=0.2, depth=0.8)

    snr = 10.0
    lambda2 = 1.0 / snr ** 2
    stc = apply_inverse(comparison, inv_default, lambda2, 'MNE')

    
    n_vertices_sample, n_times = stc.data.shape
    n_subjects = 20
    print('Simulating data for %d subjects.' % n_subjects)

    #    Let's make sure our results replicate, so set the seed.
    np.random.seed(0)
    X = randn(n_vertices_sample, n_times, n_subjects, 2) * 10
    X[:, :, :, 0] += condition1.data[:, :, np.newaxis]
    X[:, :, :, 1] += condition2.data[:, :, np.newaxis]

    # Read the source space we are morphing to
    src = mne.read_source_spaces(src)
    fsave_vertices = [s['vertno'] for s in src]
    morph_mat = mne.compute_source_morph(
        src=stc['src'], subject_to='fsaverage',
        spacing=fsave_vertices, subjects_dir="bids/").morph_mat

    n_vertices_fsave = morph_mat.shape[0]

    #    We have to change the shape for the dot() to work properly
    X = X.reshape(n_vertices_sample, n_times * n_subjects * 2)
    print('Morphing data.')
    X = morph_mat.dot(X)  # morph_mat is a sparse matrix
    X = X.reshape(n_vertices_fsave, n_times, n_subjects, 2)
    
    for face in faces:
        cars.append(face)
    del faces 
    del gafa 
    del gaca 