import readData as rd 
import mne 
from matplotlib import pyplot as plt
import os 

if __name__ == "__main__":
    ids = rd.generateIDs()

    evoked_cars_all = []
    evoked_faces_all = []
    evoked_sc_cars_all = []
    evoked_sc_faces_all = []

    for id0 in ids:
        dirName = "img/sub-{}".format(id0) 
        
        raw,evts, evtsdict = rd.readBids(id0, applyfilter=False)
        epochs = rd.get_epoched_Data(raw,evts, evtsdict)

        evokeds = dict(cars=list(epochs['cars'].iter_evoked()),
            faces=list(epochs['faces'].iter_evoked()),
            scrambled_cars=list(epochs['scrambled_cars'].iter_evoked()),
            scrambled_faces = list(epochs['scrambled_faces'].iter_evoked()))
        
        evokeds_intact = dict(cars=list(epochs['cars'].iter_evoked()),
            faces=list(epochs['faces'].iter_evoked()))

        evokeds_scarmbled = dict(scrambled_cars=list(epochs['scrambled_cars'].iter_evoked()),
            scrambled_faces = list(epochs['scrambled_faces'].iter_evoked()))
        
        comparison = mne.combine_evoked([epochs['faces'].average(), epochs['cars'].average()], [1,-1])
        n170time = comparison.copy().pick_channels(['PO8']).crop(0.11, 0.18).get_peak()[1]
        mne.viz.plot_compare_evokeds(comparison, show=False, picks=['PO8'])
        plt.savefig("img/sub-{}/sub-{}_face-car_time-{}.png".format(id0, id0, n170time))

        mne.viz.plot_compare_evokeds(evokeds_intact,show=False, ci=True, picks=['PO8'])
        plt.savefig("img/sub-{}/sub-{}_evoked_intact.png".format(id0, id0))
        mne.viz.plot_compare_evokeds(evokeds_scarmbled,show=False, picks=['PO8'], ci=True)
        plt.savefig("img/sub-{}/sub-{}_evoked_scrambled.png".format(id0, id0))
        mne.viz.plot_compare_evokeds(evokeds,show=False,  picks=['PO8'], ci=True) 
        plt.savefig("img/sub-{}/sub-{}_evoked_all.png".format(id0, id0))

        evoked_cars_all.append(epochs['cars'].average())
        evoked_faces_all.append(epochs['faces'].average())
        evoked_sc_cars_all.append(epochs['scrambled_cars'].average())
        evoked_sc_faces_all.append(epochs['scrambled_faces'].average())
    
    avg_cars = mne.grand_average(evoked_cars_all)
    avg_faces=mne.grand_average(evoked_faces_all)
    avg_sc_cars=mne.grand_average(evoked_sc_cars_all)
    avg_sc_faces=mne.grand_average(evoked_sc_faces_all)

    mne.viz.plot_compare_evokeds([avg_cars,avg_faces],show=False, ci=True, picks=['PO8'])
    plt.savefig("img/evoked_intact_all.png")
    mne.viz.plot_compare_evokeds([avg_sc_cars, avg_sc_faces],show=False, picks=['PO8'], ci=True)
    plt.savefig("img/evoked_scrambled_all.png")

    face_car = mne.combine_evoked([avg_faces, avg_cars], [1,-1])
    mne.viz.plot_compare_evokeds(face_car, picks=['PO8'],show=False)
    plt.savefig("img/evoked_intact_face-car_all.png")

    sc_face_car = mne.combine_evoked([avg_sc_faces, avg_sc_cars], [1,-1])
    mne.viz.plot_compare_evokeds(sc_face_car, picks=['PO8'],show=False)
    plt.savefig("img/evoked_intact_sc_face-car_all.png")

    face_car.pick_channels(["PO8"])
    sc_face_car.pick_channels(["PO8"])

    print(face_car.get_peak(time_as_index=False, return_amplitude=True))
    print(sc_face_car.get_peak(time_as_index=False, return_amplitude=True))
        