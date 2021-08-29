import mne
import mne_bids
import ccs_eeg_semesterproject
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd

# mappings
faces = ['stimulus/{}'.format(i) for i in range(1, 41)]
cars = ['stimulus/{}'.format(i) for i in range(41, 81)]
faces_scrambled = ['stimulus/{}'.format(i) for i in range(101, 141)]
cars_scrambled = ['stimulus/{}'.format(i) for i in range(141, 181)]


def read(subjectid, task, session, datatype, suffix, root):
    bids_path = BIDSPath(subject=subjectid, task=task, session=session,
                         datatype=datatype, suffix=suffix,
                         root=root)

    raw = read_raw_bids(bids_path)

    raw.set_montage('standard_1020', match_case=False)
    raw.load_data()
    return raw


if __name__ == "__main__":

    task = 'N170'
    suffix = 'eeg'
    session = task
    datatype = suffix
    root = "bids/n170"

    ids = ['00' + str(i) for i in range(1, 10)]
    for i in range(10, 41):
        ids.append('0' + str(i))

    df = pd.DataFrame({'subject': [], 'PO8': [], 'time': [], 'bsl': [], 'stimulus': [], 'condition': []})
    for id in ids:
        raw = read(id, task, session, datatype, suffix, root)
        raw.filter(0.5, 80, fir_design='firwin')
        raw.set_eeg_reference('average')

        ica, bad_comps = ccs_eeg_semesterproject.load_precomputed_ica(root, id, task)
        ccs_eeg_semesterproject.add_ica_info(raw, ica)
        annotations, bad_channels = ccs_eeg_semesterproject.load_precomputed_badData(root, id, task)

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
                   'response_false': 2}

        eventsdict = {'faces': 3, 'cars': 4, 'scrambled_faces': 5, 'scrambled_cars': 6, 'response_correct': 1,
                      'response_false': 2}

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

        epochs = mne.Epochs(raw, evts, eventsdict, tmin=-0.2, tmax=1, reject_by_annotation=True)

        evoked_cars = epochs['cars'].average()
        evoked_faces = epochs['faces'].average()
        evoked_scr_cars = epochs['scrambled_cars'].average()
        evoked_scr_faces = epochs['scrambled_faces'].average()
        evoked_resp_c = epochs['response_correct'].average()
        evoked_resp_f = epochs['response_false'].average()

        evoked_cars = evoked_cars.apply_baseline((None, 0))
        evoked_faces = evoked_faces.apply_baseline((None, 0))
        evoked_scr_cars = evoked_scr_cars.apply_baseline((None, 0))
        evoked_scr_faces = evoked_scr_faces.apply_baseline((None, 0))

        evoked_cars.pick_channels(['PO8'])
        evoked_faces.pick_channels(['PO8'])
        evoked_scr_cars.pick_channels(['PO8'])
        evoked_scr_faces.pick_channels(['PO8'])


        evoked_cars.crop(0.11, 0.15)
        evoked_faces.crop(0.11, 0.15)
        evoked_scr_cars.crop(0.11, 0.15)
        evoked_scr_faces.crop(0.11, 0.15)


        tmin, tmax = (-0.2, 0)
        baselines = mne.Epochs(raw, evts, eventsdict, tmin=tmin, tmax=tmax)

        carbaseline = baselines['cars'].average()
        facebaseline = baselines['faces'].average()
        scrcarbaseline = baselines['scrambled_cars'].average()
        scrfacebaseline = baselines['scrambled_faces'].average()

        cd = carbaseline.pick_channels(['PO8']).data
        fd = facebaseline.pick_channels(['PO8']).data
        sccd = scrcarbaseline.pick_channels(['PO8']).data
        scfd = scrfacebaseline.pick_channels(['PO8']).data

        scd, scf, ssccd, sscfd = (np.average(cd), np.average(fd), np.average(sccd), np.average(scfd))
        channelc, timec, ampc = evoked_cars.get_peak(time_as_index=False, return_amplitude=True)
        channelf, timef, ampf = evoked_faces.get_peak(time_as_index=False, return_amplitude=True)
        channelsc, timesc, ampsc = evoked_scr_cars.get_peak(time_as_index=False, return_amplitude=True)
        channelsf, timesf, ampsf = evoked_scr_faces.get_peak(time_as_index=False, return_amplitude=True)

        carrow = {'subject': id, 'PO8': ampc, 'time': timec, 'bsl': scd, 'stimulus': 'car', 'condition': 'intact'}
        facerow = {'subject': id, 'PO8': ampf, 'time': timef, 'bsl': scf, 'stimulus': 'face', 'condition': 'intact'}
        scrcarrow = {'subject': id, 'PO8': ampsc, 'time': timesc, 'bsl': ssccd, 'stimulus': 'car',
                     'condition': 'scrambled'}
        scrfacerow = {'subject': id, 'PO8': ampsf, 'time': timesf, 'bsl': sscfd, 'stimulus': 'face',
                      'condition': 'scrambled'}

        df = df.append(carrow, ignore_index=True)
        df = df.append(facerow, ignore_index=True)
        df = df.append(scrcarrow, ignore_index=True)
        df = df.append(scrfacerow, ignore_index=True)

    df.to_csv('erpData.csv', sep=',', index=False)
