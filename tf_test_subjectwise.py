import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_test
from mne.datasets import sample
import readData as rd 
from matplotlib import pyplot as plt

if __name__ =="__main__":
    ch_name = 'PO8'
    ids = rd.generateIDs()

    for sid in ids:
        raw, evts, eventsdict = rd.readBids(sid)
        epochs = rd.get_epoched_Data(raw, evts, eventsdict)
        epochs.load_data()

        epochs_condition_1 = epochs['faces']
        epochs_condition_1.pick_channels([ch_name])
        epochs_condition_2 = epochs['cars']
        epochs_condition_2.pick_channels([ch_name])

        decim = 2
        freqs = np.logspace(*np.log10([5, 80]), num=25)
        n_cycles = 3

        tfr_epochs_1 = tfr_morlet(epochs_condition_1, freqs,
                                n_cycles=n_cycles, decim=decim,
                                return_itc=False, average=False)

        tfr_epochs_2 = tfr_morlet(epochs_condition_2, freqs,
                                n_cycles=n_cycles, decim=decim,
                                return_itc=False, average=False)

        tfr_epochs_1.apply_baseline(mode='ratio', baseline=(None, 0))
        tfr_epochs_2.apply_baseline(mode='ratio', baseline=(None, 0))

        epochs_power_1 = tfr_epochs_1.data[:, 0, :, :]  # only 1 channel as 3D matrix
        epochs_power_2 = tfr_epochs_2.data[:, 0, :, :]  # only 1 channel as 3D matrix

        threshold = 6.0
        T_obs, clusters, cluster_p_values, H0 = \
            permutation_cluster_test([epochs_power_1, epochs_power_2], out_type='mask',
                                     n_permutations=1000, threshold=threshold, tail=0)

        
        times = 1e3 * epochs_condition_1.times  # change unit to ms
        evoked_condition_1 = epochs_condition_1.average()
        evoked_condition_2 = epochs_condition_2.average()

        plt.figure()
        plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)

        plt.subplot(2, 1, 1)
        # Create new stats image with only significant clusters
        T_obs_plot = np.nan * np.ones_like(T_obs)
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= 0.05:
                T_obs_plot[c] = T_obs[c]

        plt.imshow(T_obs,
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                aspect='auto', origin='lower', cmap='gray')
        plt.imshow(T_obs_plot,
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                aspect='auto', origin='lower', cmap='RdBu_r')

        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Induced power (%s)' % ch_name)

        ax2 = plt.subplot(2, 1, 2)
        evoked_contrast = mne.combine_evoked([evoked_condition_1, evoked_condition_2],
                                             weights=[1, -1])
        evoked_contrast.plot(axes=ax2, time_unit='s', show=False)

        plt.savefig("img/sub-{}/sub-{}_TF_permutationCluster.png".format(sid, sid))