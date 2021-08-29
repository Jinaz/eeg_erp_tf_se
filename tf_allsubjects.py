import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_test
from mne.datasets import sample
import readData as rd 
from matplotlib import pyplot as plt

def generatePowerAndInduced(epochs):
    freqs = np.logspace(*np.log10([5, 80]), num=25)
    n_cycles = freqs/2
    power_total = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=4,average=True)

    epochs_induced = epochs.copy()
    epochs_induced.subtract_evoked()

    power_induced = mne.time_frequency.tfr_morlet(epochs_induced, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=4,average=True)

    return power_total, power_induced

def pltEvoked(power_total, power_induced, condition, vmin=-1, vmean=0.0000000015):
    baseline = (None, 0)
    evoked = mne.combine_evoked([power_total, power_induced], [1,-1])
    vmin = vmin
    vmean = vmean
    pltandsave(evoked, power_total, power_induced, baseline, vmin, vmean, condition)

    return evoked 

def pltandsave(evoked,power_total, power_induced, baseline, vmin, vmean, condition):
    power_total.plot(baseline=baseline, mode="percent", picks='PO8', vmin=vmin, vmax=-vmin, show=False)
    plt.savefig("img/tf-{}_powertotal.png".format(condition))
    power_induced.plot(baseline=baseline, mode="percent", picks='PO8', vmin=vmin, vmax=-vmin, show=False)
    plt.savefig("img/tf-{}_powerinduced.png".format(condition))
    evoked.plot(baseline=baseline, mode="mean", picks='PO8', vmin=-vmean, vmax=vmean, show=False)
    plt.savefig("img/tf-{}-evoked.png".format(condition))

if __name__ =="__main__":
    ch_name = 'PO8'
    ids = rd.generateIDs()

    face = []
    car = []
    faceinduced = []
    carinduced = []

    scface = []
    sccar = []
    scfaceinduced = []
    sccarinduced = []

    decim = 2
    freqs = np.logspace(*np.log10([5, 80]), num=25)
    n_cycles = freqs/2

    conditions = ['faces', 'cars', 'scrambled_faces', 'scrambled_cars']

    for sid in ids:
        raw, evts, eventsdict = rd.readBids(sid)
        epochs = rd.get_epoched_Data(raw, evts, eventsdict)
        epochs.load_data()
        for condition in conditions:
            epochs_condition = epochs[condition]
            power, induced = generatePowerAndInduced(epochs_condition)

            if condition == 'faces':
                face.append(power)
                faceinduced.append(induced)
            elif condition == 'cars':
                car.append(power)
                carinduced.append(induced)
            elif condition == 'scrambled_faces':
                scface.append(power)
                scfaceinduced.append(induced)
            elif condition == 'scrambled_cars':
                sccar.append(power)
                sccarinduced.append(induced)
        

    face = mne.grand_average(face)
    car = mne.grand_average(car)
    faceinduced = mne.grand_average(faceinduced)
    carinduced = mne.grand_average(carinduced)

    scface = mne.grand_average(scface)
    sccar = mne.grand_average(sccar)
    scfaceinduced = mne.grand_average(scfaceinduced)
    sccarinduced = mne.grand_average(sccarinduced)

    evoked_comparison_face_car = mne.combine_evoked([face, car], [1,-1])
    evoked_comparison_face_car_induced = mne.combine_evoked([faceinduced, carinduced], [1,-1])

    pltEvoked(face, faceinduced, "all-faces")
    pltEvoked(car, carinduced, "all-cars")
    pltEvoked(evoked_comparison_face_car, evoked_comparison_face_car_induced, "all-face-car")