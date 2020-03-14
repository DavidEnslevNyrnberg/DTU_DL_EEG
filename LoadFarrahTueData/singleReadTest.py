import os, itertools, json, datetime, mne
from collections import defaultdict
# from NovelEEG.SampleCode import loadFromJson
import loadFromJson
import numpy as np

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\" # ~~~~~~ CHANGE THIS TO YOUR DIR-PATH
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"
farrahDataDir = saveDir + farrahData

# load json to dict
# loadFromJson()
edfDefDict = loadFromJson.loadByJson()
# print(edfDefDict)

############ TODO: proof of concepts'
print('done now to proof of concepts')

### plots and load of several and single .edf file
# loading several files by [list]
from mne.io import read_raw_edf
sevFiles = ["sbs2data_2018_09_01_08_04_51_328.edf",
             "sbs2data_2018_09_01_08_36_16_331.edf",
             "sbs2data_2018_09_01_08_56_49_330.edf",
             "sbs2data_2018_09_01_09_08_16_335.edf",
             "sbs2data_2018_09_01_09_44_21_337.edf"]
edfDict = defaultdict(dict)
for i in sevFiles:
    edfDict[i] = edfDefDict[i].copy()
    edfDict[i]["path"] = saveDir+edfDict[i]["path"][0]
    edfDict[i]["rawData"] = read_raw_edf(edfDict[i]["path"], preload=True, stim_channel='auto')
    edfDict[i]["t0"] = datetime.datetime.fromtimestamp(edfDict[i]["rawData"].annotations.orig_time - 60 * 60)
    edfDict[i]["tN"] = edfDict[i]["rawData"].annotations.orig_time
# TODO: set correct start and end time

# same method for loading single file
singFile = ["sbs2data_2018_09_07_16_41_25_457.edf"]
edfDict2 = defaultdict(dict)
for i in singFile:
    edfDict2[i] = edfDefDict[i].copy()
    edfDict2[i]["path"] = saveDir+edfDict2[i]["path"][0] # if json file doesn't have your development dir
    edfDict2[i]["rawData"] = read_raw_edf(edfDict2[i]["path"], preload=True, stim_channel='auto')

# pre-processing single file
# band-pass filter [1hz; 30hz]
foo = edfDict2["sbs2data_2018_09_07_16_41_25_457.edf"]["rawData"].copy()
foo.plot_psd()
foo.filter(1, 30, fir_design='firwin')
foo.plot_psd()
foo.set_eeg_reference()

# debug mode
# edfDefDict.clear()
# edfDict.clear()
# edfDict2.clear()

# pre-processing single file
def pipeline(EEGseries=None, lpfq=1, hpfq=45):
    EEGseries.filter(lpfq, hpfq, fir_design='firwin')
    EEGseries.set_eeg_reference()
    return EEGseries
# band-pass filter [1hz; 30hz]
for edf in edfDict:
    print(edf)
    EEGseries = pipeline(edfDict[edf]["rawData"])

foo = edfDict.copy()
foo.plot_psd()
foo.filter(1, 30, fir_design='firwin')
foo.plot_psd()
foo.set_eeg_reference()






# explore raw file
edfRaw = edfDict2[singFile[0]]["rawData"].copy()
edfRaw.info
edfRaw.annotations.orig_time
datetime.datetime.fromtimestamp(edfRaw.annotations.orig_time-60*60) # to get datetime object of orig_time

# TODO: check montage method - does it mean-average channels - how does channels relate to each other
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
f, t, Sxx = signal.spectrogram(edfRaw.get_data()[-1, t0:(t0+tWindow)*128], fs=128)
plt.pcolormesh(t, f, np.log(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

