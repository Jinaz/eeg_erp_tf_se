import readData as rd 
import mne 
from matplotlib import pyplot as plt 
import numpy as np

def generatePowerAndInduced(epochs):
    freqs = np.logspace(*np.log10([5, 80]), num=25)
    n_cycles = freqs/2 
    power_total = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=4,average=True)

    epochs_induced = epochs.copy()
    epochs_induced.subtract_evoked()

    power_induced = mne.time_frequency.tfr_morlet(epochs_induced, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=4,average=True)

    return power_total, power_induced

def pltEvoked(power_total, power_induced,sid, condition, vmin=-1, vmean=0.0000000015):
    baseline = (None, 0)
    evoked = mne.combine_evoked([power_total, power_induced], [1,-1])
    vmin = vmin
    vmean = vmean
    pltandsave(evoked, power_total, power_induced, baseline, vmin, vmean, sid, condition)

    return evoked 

def pltandsave(evoked,power_total, power_induced, baseline, vmin, vmean, sid, condition):
    power_total.plot(baseline=baseline, mode="percent", picks='PO8', vmin=vmin, vmax=-vmin, show=False)
    plt.savefig("img/sub-{}/sub-{}_tf-{}_powertotal.png".format(sid,sid,condition))
    power_induced.plot(baseline=baseline, mode="percent", picks='PO8', vmin=vmin, vmax=-vmin, show=False)
    plt.savefig("img/sub-{}/sub-{}_tf-{}_powerinduced.png".format(sid,sid,condition))
    evoked.plot(baseline=baseline, mode="mean", picks='PO8', vmin=-vmean, vmax=vmean, show=False)
    plt.savefig("img/sub-{}/sub-{}_tf-{}-evoked.png".format(sid,sid,condition))

if __name__ == "__main__":
    subjects = rd.generateIDs()
    
    
    for sid in subjects:
        raw, evts, eventsdict = rd.readBids(sid)
        epochs = rd.get_epoched_Data(raw, evts, eventsdict)

        epochs_faces = epochs['faces']
        epochs_cars = epochs['cars']

        epochs_scrambled_faces = epochs['scrambled_faces']
        epochs_scrambled_cars = epochs['scrambled_cars']

        power_total_faces, power_induced_faces = generatePowerAndInduced(epochs_faces)
        power_total_cars, power_induced_cars = generatePowerAndInduced(epochs_cars)
        power_total_scrambled_faces, power_induced_scrambled_faces = generatePowerAndInduced(epochs_scrambled_faces)
        power_total_scrambled_cars, power_induced_scrambled_cars = generatePowerAndInduced(epochs_scrambled_cars)

        evoked_faces = pltEvoked(power_total_faces, power_induced_faces, sid, "faces", vmean=0.0000000015)
        evoked_cars = pltEvoked(power_total_cars, power_induced_cars, sid, "cars")
        evoked_scrambled_faces = pltEvoked(power_total_scrambled_faces, power_induced_scrambled_faces, sid, "scrambled-faces")
        evoked_scrambled_cars = pltEvoked(power_total_scrambled_cars, power_induced_scrambled_cars, sid, "scrambled-cars")


        power_allfaces = mne.grand_average([power_total_faces, power_total_scrambled_faces])
        power_allcars = mne.grand_average([power_total_cars, power_total_scrambled_cars])
        power_induced_allfaces = mne.grand_average([power_induced_faces, power_induced_scrambled_faces])
        power_induced_allcars = mne.grand_average([power_induced_cars, power_induced_scrambled_cars])

        power_face_car = mne.combine_evoked([power_total_faces, power_total_cars], weights=[1, -1])
        power_face_car_induced = mne.combine_evoked([power_induced_faces, power_induced_cars], weights=[1, -1])
        pltEvoked(power_face_car, power_face_car_induced,sid, "intact-face-car", vmin=-1, vmean=0.0000000015)

        power_all_face_car= mne.combine_evoked([power_allfaces, power_allcars], weights=[1, -1])
        power_all_face_car_induced= mne.combine_evoked([power_induced_allfaces, power_induced_allcars], weights=[1, -1])
        pltEvoked(power_all_face_car, power_all_face_car_induced,sid, "all-face-car", vmin=-1, vmean=0.0000000015)

        
        
        





        