import os, mne
from collections import defaultdict
from datetime import datetime
from mne.io import read_raw_edf
import numpy as np
# from NovelEEG.SampleCode.loadData import jsonLoad
from loadData import jsonLoad
# from testEnv import readRawEdf

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\" # ~~~ What is your execute path?
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\" # ~~~ What is the name of your data folder?
jsonDir = r"edfFiles.json" # ~~~ Where is your json folder?
jsonDataDir = saveDir + jsonDir
farrahDataDir = saveDir + farrahData

# pre-processing pipeline single file
def pipeline(EEGseries=None, lpfq=1, hpfq=40, notchfq=50):
    EEGseries.plot
    EEGseries.set_montage(mne.channels.read_montage(kind='easycap-M1', ch_names=EEGseries.ch_names))
    EEGseries.plot_psd()
    EEGseries.notch_filter(freqs=notchfq, notch_widths=5)
    EEGseries.plot_psd()
    EEGseries.filter(lpfq, hpfq, fir_design='firwin')
    EEGseries.set_eeg_reference()
    EEGseries.plot_sensors(show_names=True)
    return EEGseries


# TODO: set correct tN
def readRawEdf(edfDict=None, read_raw_edf_param={'preload':True, 'stim_channel':'auto'}):
    edfDict["rawData"] = read_raw_edf(saveDir+edfDict["path"][0], **read_raw_edf_param)
    edfDict["t0"] = datetime.fromtimestamp(edfDict["rawData"].annotations.orig_time - 60*60)
    # edfDict["tN"] = "mandag fuck o'clock"
    # edfDict["t.step"] = "00:20"
    # edfDict["t.window"] = "02:00"
    return edfDict


# load json to dict
edfDefDict = jsonLoad(path=jsonDataDir)

############ TODO: proof of concepts'
# loading several files by [list]
# fileSeveral = ["sbs2data_2018_09_01_08_04_51_328.edf",
#              "sbs2data_2018_09_01_08_36_16_331.edf",
#              "sbs2data_2018_09_01_08_56_49_330.edf",
#              "sbs2data_2018_09_01_09_08_16_335.edf",
#              "sbs2data_2018_09_01_09_44_21_337.edf"]
# edfDict = defaultdict(dict)
# for edf in fileSeveral:
#     edfDict[edf] = edfDefDict[edf].copy()
#     edfDict[edf] = readRawEdf(edfDict[edf])
#     pipeline(edfDict[edf]["rawData"])
    # edfDict[edf]["rawData"].plot_psd()

# same method for loading single file
fileSingle = ["sbs2data_2018_09_07_16_41_25_457.edf"]
edfDict2 = defaultdict(dict)
for edf in fileSingle:
    edfDict2[edf] = edfDefDict[edf].copy()
    edfDict2[edf] = readRawEdf(edfDict2[edf])
    pipeline(edfDict2[edf]["rawData"])
    edfDict2[edf]["rawData"].plot_psd()

# # pre-processing single file
# # band-pass filter [1Hz; 40Hz]
# # notch filter [50Hz +- 5Hz]
# foo = edfDict2["sbs2data_2018_09_07_16_41_25_457.edf"]["rawData"].copy()
# # foo.plot_psd()
# foo.filter(1, 40, fir_design='firwin')
# # foo.plot_psd()
# foo.set_eeg_reference()

# debug mode
# edfDefDict.clear()
# edfDict.clear()
# edfDict2.clear()


# for edf in edfDict:
#     print(edf)
#     pipeline(edfDict[edf]["rawData"])
#     edfDict[edf]["rawData"].plot_psd()

# foo = edfDict.copy()
# foo.plot_psd()
# foo.filter(1, 40, fir_design='firwin')
# foo.plot_psd()
# foo.set_eeg_reference()


"""
This is where I left off
Loading and pipe line is now functions
but generation of spectrograms is still cluncky/wrong

start with reading from here ;)
"""

# explore raw file
edfRaw = edfDict2[fileSingle[0]]["rawData"].copy()
edfRaw.info
edfRaw.annotations.orig_time
datetime.fromtimestamp(edfRaw.annotations.orig_time-60*60) # to get datetime object of orig_time

# TODO: check montage method
testMont = mne.channels.read_montage(kind='easycap-M1', ch_names=edfRaw.ch_names)
edfRaw.set_montage(testMont)
edfRaw.plot_sensors(show_names=True)
edfRaw.plot_psd()
edfRaw.plot()


# !!!this is not functional yet!!!
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
t0 = 0
tWindow = 60*2
# TODO: read there is return_times, check how it works
(fooTest, timeTest) = edfRaw.get_data(start=t0, stop=(tWindow+t0)*128, return_times=True)

f, t, Sxx = signal.spectrogram(edfRaw.get_data()[-1, t0:(t0+tWindow)*128], fs=128)
plt.pcolormesh(t, f, np.log(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#### TODO: ANDERS CODE
""" Cutting intervals and creating spectrograms """
# Overlap in time intervals (in seconds; time intervals are 2 minutes long)
overlap = 60

dataset = []
dataset_labels = []
counter = 0
for item in edfDict2.items():
    if counter >= 20:  # Limit max dataset for now
        break
    raw = mne.io.read_raw_edf(item[1]['path'][0])

    # set up and fit the ICA -- NOT IMPLEMENTED (THIS IS FROM TUTORIAL)

    # ica = mne.preprocessing.ICA(n_components=14, random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.exclude = [1, 2]  # details on how we picked these are omitted here
    # ica.plot_properties(raw, picks=ica.exclude)
    # Removing components below:
    # orig_raw = raw.copy()
    raw.load_data()
    # ica.apply(raw)

    # show some frontal channels to clearly illustrate the artifact removal
    chs = ['TP10', 'Fz', 'P3', 'Cz', 'C4', 'TP9', 'Pz', 'P4', 'FT7',
           'C3', 'O1', 'FT8', 'Fpz', 'O2']
    # chan_idxs = [raw.ch_names.index(ch) for ch in chs]
    # orig_raw.plot(order=chan_idxs, start=12, duration=4)
    # raw.plot(order=chan_idxs, start=12, duration=4)

    # only get Sxx:
    patient = []
    for j in range(5):
        images = []
        image_labels = []
        for i in range(len(raw["data"][0])):
            # range(len(raw["data"][0][i])//sample_freq):
            f, t, Sxx = spectrogram(
                np.array(raw["data"][0][i][j * overlap * sample_freq:(j * overlap + 120) * sample_freq]).flatten(),
                sample_freq)
            images.append(np.log(Sxx + 10e-20))
            image_labels.append(np.array([counter, i, j]))
        dataset.append(images)
        dataset_labels.append(image_labels)
    # dataset.append(patient)
    counter += 1