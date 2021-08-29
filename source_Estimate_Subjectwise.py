import os.path as op
import mne
from mne.datasets import fetch_fsaverage
import readData as rd
from matplotlib import pyplot as plt 
from surfer import Brain
from mne.minimum_norm import make_inverse_operator, apply_inverse

def get_epochs(epochs):
    
    
    int_faces = epochs['faces']
    int_cars= epochs['cars']
    sc_faces= epochs['scrambled_faces']
    sc_cars= epochs['scrambled_cars']

    int_faces.set_eeg_reference(projection=True)
    int_cars.set_eeg_reference(projection=True)
    sc_faces.set_eeg_reference(projection=True)
    sc_cars.set_eeg_reference(projection=True)

        


    return int_faces, int_cars, sc_faces, sc_cars

def savePlots(stc, inittime, sid, conditionname):
    h = mne.viz.plot_source_estimates(stc, initial_time=inittime, time_viewer=False,time_unit='s',views="lateral",hemi='split', size=(800, 400), show_traces=False)
    img = stc_plot2img(h,closeAfterwards=True)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("img/sub-{}/sub-{}_SourceEstimateAt-{}_C-{}.png".format(sid, sid, inittime,conditionname))


def sourceEstimatePipeline(epochs, epochslist, evokeds, trans, src, bem, timestart, timeend):
    inittime = evokeds.copy().crop(timestart,timeend).pick(['PO8']).get_peak()[1]

    fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0)
    noise_cov = mne.compute_covariance(epochslist, tmax=0)
    #noise_cov.plot(epochs.info)

    inv_default = make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)

    snr = 10.0
    lambda2 = 1.0 / snr ** 2
    stc = apply_inverse(evokeds, inv_default, lambda2, 'MNE')
    return stc, inittime

def stc_plot2img(h,title="SourceEstimate",closeAfterwards=False,crop=True):
    h.add_text(0.1, 0.9, title, 'title', font_size=16)
    screenshot = h.screenshot()
    if closeAfterwards:
        h.close()

    if crop:
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    return screenshot

if __name__ == "__main__":

    fsdir = fetch_fsaverage()
    subjectsdir = op.dirname(fsdir)

    subject = 'fsaverage'
    trans = 'fsaverage'
    src = op.join(fsdir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fsdir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    

    subjectids = rd.generateIDs()
    for sid in subjectids:

        raw, evts, evtsdict = rd.readBids(sid)
        epochs = rd.get_epoched_Data(raw, evts, evtsdict)
        epochs_faces, epochs_cars, sc_faces, sc_cars = get_epochs(epochs)
        #for combined

        epochs_cars.set_eeg_reference(projection=True)
        epochs_faces.set_eeg_reference(projection=True)

        evfaces = epochs_faces.average()
        evcars = epochs_cars.average()


        comparison = mne.combine_evoked([evfaces,evcars], [1,-1])
        stc, inittime = sourceEstimatePipeline(epochs, [epochs_faces, epochs_cars], comparison, trans, src, bem, 0.11, 0.24)
        savePlots(stc, inittime, sid, "face-car")

        stc, inittime = sourceEstimatePipeline(epochs, epochs_faces, evfaces, trans, src, bem, 0.11, 0.18)
        savePlots(stc, inittime, sid, "faces")

        stc, inittime = sourceEstimatePipeline(epochs, epochs_cars, evcars, trans, src, bem, 0.11, 0.24)
        savePlots(stc, inittime, sid, "cars")