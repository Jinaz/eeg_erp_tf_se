import mne 
import mne_bids
import ccs_eeg_semesterproject
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import scipy
from matplotlib import pyplot as plt 

task = 'N170'
suffix = 'eeg'
session = task
datatype = suffix
root = "bids/n170"

def generateIDs():
    ids = ['00' + str(i) for i in range(1, 10)]
    for i in range(10, 41):
        ids.append('0' + str(i))
    return ids 

faces = ['stimulus/{}'.format(i) for i in range(1,41)]
cars = ['stimulus/{}'.format(i) for i in range(41,81)]
faces_scrambled = ['stimulus/{}'.format(i) for i in range(101,141)]
cars_scrambled = ['stimulus/{}'.format(i) for i in range(141, 181)]

def readBids(subjectid, applyfilter=True):
    bids_path = BIDSPath(subject=subjectid, task=task, session=session,
                         datatype=datatype, suffix=suffix,
                         root=root)

    # read the file
    raw = read_raw_bids(bids_path)

    raw.set_channel_types({'HEOG_left':'eog', 'HEOG_right':'eog', 'VEOG_lower':'eog'})

    raw.set_montage('standard_1020', match_case=False)
    raw.load_data()

    if applyfilter:
        raw = raw.filter(0.5, 80, fir_design='firwin')
    raw.set_eeg_reference('average')

    ica, bad_comps = ccs_eeg_semesterproject.load_precomputed_ica(root, subjectid, task)
    ccs_eeg_semesterproject.add_ica_info(raw, ica)
    annotations, bad_channels = ccs_eeg_semesterproject.load_precomputed_badData(root, subjectid, task)

    for annot in annotations:
        raw.annotations.append(annot['onset'], annot['duration'], annot['description'])

    raw.interpolate_bads()

    ica.apply(raw, exclude=bad_comps)


    stimuluskey = 'stimulus'
    responsekey = 'response'

    evts, evtsdict = mne.events_from_annotations(raw)

    carkeys = [evtsdict.get(i) for i in evtsdict.keys() if i in cars]
    facekeys = [evtsdict.get(i) for i in evtsdict.keys() if i in faces]

    scrcarkeys = [evtsdict.get(i) for i in evtsdict.keys() if i in cars_scrambled]
    scrfacekeys = [evtsdict.get(i) for i in evtsdict.keys() if i in faces_scrambled]


    eventid = {'faces': {i for i in facekeys},
           'cars': {i for i in carkeys},
           'faces_scrambled': {i for i in scrfacekeys},
           'cars_scrambled': {i for i in scrcarkeys},
           'response_correct': 1,
           'response_false':2}

    eventsdict = {'faces':3, 'cars':4, 'scrambled_faces':5, 'scrambled_cars':6, 'response_correct':1, 'response_false':2}

    for evt in evts:
        if evt[2] in eventid['faces']:
            evt[2] = 3
        elif evt[2] in eventid['cars']:
            evt[2] = 4
        elif evt[2] in eventid['faces_scrambled']:
            evt[2] = 5
        elif evt[2] in eventid['cars_scrambled']:
            evt[2] = 6
        elif evt[2] == eventid['response_correct']:
            evt[2] = 1
        elif evt[2] == eventid['response_false']:
            evt[2] = 2

    return raw, evts, eventsdict 


def get_epoched_Data(raw, evts, eventsdict, rejectByAnnotation=True):
    epochs = mne.Epochs(raw, evts, eventsdict, tmin=-0.2, tmax=1, reject_by_annotation=rejectByAnnotation)
    
    return epochs

def getPowers(epochs, freqs= np.logspace(*np.log10([4,80]), num=30),applybsl=False, baseline=(None, 0), itc=False):
    epochs_induced = epochs.copy()
    epochs_induced.subtract_evoked()
    epcopy = epochs.copy()

    ep_cars = epcopy['cars']
    ep_faces = epcopy['faces']
    ep_sc_cars = epcopy['scrambled_cars']
    ep_sc_faces = epcopy['scrambled_faces']

    epochs_induced_cars = epochs_induced['cars']
    epochs_induced_faces = epochs_induced['faces']
    epochs_induced_sc_cars = epochs_induced['scrambled_cars']
    epochs_induced_sc_faces = epochs_induced['scrambled_faces']

    n_cycles = freqs /2
    n_jobs = 8
    if itc:
        power_cars, itc_cars = mne.time_frequency.tfr_morlet(ep_cars, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)
        power_faces, itc_faces = mne.time_frequency.tfr_morlet(ep_faces, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)
        power_sc_cars, itc_sc_cars = mne.time_frequency.tfr_morlet(ep_sc_cars, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)
        power_sc_faces, itc_sc_faces = mne.time_frequency.tfr_morlet(ep_sc_faces, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)

        power_induced_cars, itc_induced_cars = mne.time_frequency.tfr_morlet(epochs_induced_cars, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)
        power_induced_faces, itc_induced_faces = mne.time_frequency.tfr_morlet(epochs_induced_faces, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)
        power_induced_sc_cars, itc_induced_sc_cars = mne.time_frequency.tfr_morlet(epochs_induced_sc_cars, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)
        power_induced_sc_faces, itc_induced_sc_faces = mne.time_frequency.tfr_morlet(epochs_induced_sc_faces, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=itc)

    else:
        power_cars = mne.time_frequency.tfr_morlet(ep_cars, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=n_jobs,average=True)
        power_faces = mne.time_frequency.tfr_morlet(ep_faces, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=n_jobs,average=True)
        power_sc_cars = mne.time_frequency.tfr_morlet(ep_sc_cars, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=n_jobs,average=True)
        power_sc_faces = mne.time_frequency.tfr_morlet(ep_sc_faces, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=n_jobs,average=True)

        power_induced_cars = mne.time_frequency.tfr_morlet(epochs_induced_cars, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=False)
        power_induced_faces = mne.time_frequency.tfr_morlet(epochs_induced_faces, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=False)
        power_induced_sc_cars = mne.time_frequency.tfr_morlet(epochs_induced_sc_cars, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=False)
        power_induced_sc_faces = mne.time_frequency.tfr_morlet(epochs_induced_sc_faces, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,average=True, return_itc=False)

    if applybsl:
        power_cars.apply_baseline(mode='ratio', baseline=baseline)
        power_faces.apply_baseline(mode='ratio', baseline=baseline)
        power_sc_cars.apply_baseline(mode='ratio', baseline=baseline)
        power_sc_faces.apply_baseline(mode='ratio', baseline=baseline)
    if itc:
        return power_cars, power_faces, power_sc_cars, power_sc_faces, itc_cars, itc_faces, itc_sc_cars, itc_sc_faces, power_induced_cars, power_induced_faces, power_induced_sc_cars,power_induced_sc_faces, itc_induced_cars, itc_induced_faces, itc_induced_sc_cars, itc_induced_sc_faces
    else:
        return power_cars, power_faces, power_sc_cars, power_sc_faces, power_induced_cars, power_induced_faces, power_induced_sc_cars, power_induced_sc_faces


