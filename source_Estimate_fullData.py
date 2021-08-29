import os.path as op
import mne
from mne.datasets import fetch_fsaverage
import readData as rd
from matplotlib import pyplot as plt 
from surfer import Brain
from mne.minimum_norm import make_inverse_operator, apply_inverse
import source_Estimate as ses 

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

def savePlots(stc, inittime, conditionname):
    h = mne.viz.plot_source_estimates(stc, initial_time=inittime, time_viewer=False,time_unit='s',views="lateral",hemi='split', size=(800, 400), show_traces=False)
    img = stc_plot2img(h,closeAfterwards=True)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("img/all_SourceEstimateAt-{}_C-{}.png".format(inittime,conditionname))

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
    
    stc, inittime = ses.sourceEstimatePipeline(epochsinfo, faces, gafa, trans, src, bem, 0.11, 0.18)
    savePlots(stc, inittime, "face-ALL")

    stc, inittime = ses.sourceEstimatePipeline(epochsinfo, cars, gaca, trans, src, bem, 0.11, 0.24)
    savePlots(stc, inittime, "car-ALL")

    

    for face in faces:
        cars.append(face)
    del faces 
    
    comparison = mne.combine_evoked([gafa, gaca], [1,-1])
    del gafa 
    del gaca 


    stc, inittime = ses.sourceEstimatePipeline(epochsinfo, cars, comparison, trans, src, bem, 0.11, 0.24)
    savePlots(stc, inittime, "face-car-ALL")