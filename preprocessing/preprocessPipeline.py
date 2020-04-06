import os, mne, torch
from collections import defaultdict
from datetime import datetime
from mne.io import read_raw_edf
import numpy as np
# from LoadFarrahTueData.loadData import jsonLoad, pathLoad
from loadData import jsonLoad
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\ander\Documents\DTU_data_EEG"+"\\"  # ~~~ What is your execute path?
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"  # ~~~ What is the name of your data folder?
jsonDir = r"SampleCode\edfFiles_AndersremovedIncorrectChannelRec.json" # ~~~ Where is your json folder?
jsonDataDir = saveDir + jsonDir
farrahDataDir = saveDir + farrahData

# class preprocessPipeline():
#     def __init__(self, saveDir=os.getcwd(), loadDir=None):
#         if not os.path.exists:
#             os.mkdir(saveDir)
#         if loadDir.split(".")[-1] == "json":
#             edfDict = jsonLoad(saveDir=saveDir, loadDir=loadDir)
#         else:
#             edfDict = pathLoad(path=loadDir)

def readRawEdf(edfDict=None, read_raw_edf_param={'preload':True, 'stim_channel':'auto'}, tWindow=120, tStep=30):
    edfDict["rawData"] = read_raw_edf(saveDir+edfDict["path"][0], **read_raw_edf_param)
    tStart = edfDict["rawData"].annotations.orig_time - 60*60
    tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
    edfDict["t0"] = datetime.fromtimestamp(tStart)
    edfDict["tN"] = datetime.fromtimestamp(tStart + tLast),
    edfDict["tWindow"] = tWindow
    edfDict["tStep"] = tStep
    edfDict["fS"] = edfDict["rawData"].info["sfreq"]
    return edfDict

# pre-processing pipeline single file
def pipeline(EEGseries=None, lpfq=1, hpfq=40, notchfq=50):
    # EEGseries.plot
    EEGseries.set_montage(mne.channels.read_montage(kind='easycap-M1', ch_names=EEGseries.ch_names))
    # EEGseries.plot_psd()
    EEGseries.notch_filter(freqs=notchfq, notch_widths=5)
    # EEGseries.plot_psd()
    EEGseries.filter(lpfq, hpfq, fir_design='firwin')
    EEGseries.set_eeg_reference()
    # EEGseries.plot_sensors(show_names=True)
    return EEGseries

def spectrogramMake(EEGseries=None, t0=0, tWindow=120):
    edfFs = EEGseries.info["sfreq"]
    chWindows = EEGseries.get_data(start=int(t0), stop=int(t0+tWindow))
    _, _, Sxx = signal.spectrogram(chWindows, fs=edfFs)
    # fTemp, tTemp, Sxx = signal.spectrogram(chWindows, fs=edfFs)
    # plt.pcolormesh(tTemp, fTemp, np.log(Sxx))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title("channel spectrogram: "+EEGseries.ch_names[count])
    # plt.show()
    return torch.tensor(np.log(Sxx+np.finfo(float).eps)) # for np del torch.tensor

def slidingWindow(edfInfo=None, tN=0, tStep=60, sample_freq = 128, last_frac=0, localSave={"sliceSave":False, "saveDir":os.getcwd()}):
    # windowEEG = defaultdict(list)
    sampleWindow = edfInfo["tWindow"]*edfInfo["fS"]
    for i in range(0, int(tN-sampleWindow), int(tStep*sample_freq)):
        # windowKey = "window_%i_%i" % (i, i+sampleWindow)
        # windowEEG[windowKey] = spectrogramMake(edfInfo["rawData"], t0=i, tWindow=sampleWindow)
        cut = spectrogramMake(edfInfo["rawData"], t0=i, tWindow=sampleWindow)
        if localSave["sliceSave"]:
            idDir = edfInfo["rawData"].filenames[0].split('\\')[-1].split('.')[0]
            if not os.path.exists(saveDir + "tempData\\"):
                os.mkdir(saveDir + "tempData\\")
            torch.save(cut, saveDir + "tempData\\%s.pt" % (idDir+'-'+str(i))) # for np del torch.tensor

    if (1+tN) % int(sampleWindow) > sampleWindow*last_frac:
        # windowKey = "window_%i_%i" % (int(tN-sampleWindow), int(tN))
        # windowEEG[windowKey] = spectrogramMake(edfInfo["rawData"], t0 = int(tN-sampleWindow), tWindow = sampleWindow)
        cut = spectrogramMake(edfInfo["rawData"], t0 = int(tN-sampleWindow), tWindow = sampleWindow)
        if localSave["sliceSave"]:
            idDir = edfInfo["rawData"].filenames[0].split('\\')[-1].split('.')[0]
            if not os.path.exists(saveDir + "tempData\\"):
                os.mkdir(saveDir + "tempData\\")
            torch.save(cut, saveDir + "tempData\\%s.pt" % (idDir+'-'+str(1+tN))) # for np del torch.tensor
    if not localSave["sliceSave"]:
        windowOut = windowEEG.copy()
    else:
        windowOut = None
    return windowOut

def completePrep(tWin=120, tStep=60, sample_freq = 128, last_frac=0, localSave={"sliceSave":False, "saveDir":os.getcwd()}, # ANDERS ændret tstep fra 30 til 60
                 notchFQ=50, lpFQ=1, hpFQ=40):
    for edf in tqdm(edfDict.keys()):
        edfDict[edf] = readRawEdf(edfDict[edf], tWindow=tWin, tStep=tStep)
        pipeline(edfDict[edf]["rawData"], lpfq=lpFQ, hpfq=hpFQ, notchfq=notchFQ)
        tLastN = edfDict[edf]["rawData"].last_samp
        slidingWindow(edfInfo=edfDict[edf], tN=tLastN, tStep=tStep, sample_freq=sample_freq,
                      last_frac=last_frac, localSave=localSave)
        # windowDict =
        # edfDict[edf]["tensors"] = windowDict
    return edfDict

# for fun try with andreas


# load json to dict
edfDefDict = jsonLoad(path=jsonDataDir)

# load ALL .edf paths
edfDict = edfDefDict.copy()
for edf in edfDict:
    edfDict[edf] = readRawEdf(edfDict[edf])
    # preprocessPipeline.pipeline(edfDict[edf]["rawData"])
    # edfDict[edf]["rawData"].plot_psd()

# Preprocess and save entire dataset locally
completePrep(localSave={"sliceSave":True, "saveDir":0})




# Old / individual preprocessing

# loading several files by [list]
"""
fileSeveral = ["sbs2data_2018_09_01_08_04_51_328.edf",
             "sbs2data_2018_09_01_08_36_16_331.edf",
             "sbs2data_2018_09_01_08_56_49_330.edf",
             "sbs2data_2018_09_01_09_08_16_335.edf",
             "sbs2data_2018_09_01_09_44_21_337.edf"]
edfDict = defaultdict(dict)
for edf in fileSeveral:
    edfDict[edf] = edfDefDict[edf].copy()
    edfDict[edf] = readRawEdf(edfDict[edf])
    pipeline(edfDict[edf]["rawData"])
    # edfDict[edf]["rawData"].plot_psd()
"""
# same method for loading single file
# fileSingle = ["sbs2data_2018_09_07_16_41_25_457.edf"]
# edfDict2 = defaultdict(dict)
# for edf in fileSingle:
#     edfDict2[edf] = edfDefDict[edf].copy()
#     edfDict2[edf] = readRawEdf(edfDict2[edf])
#     pipeline(edfDict2[edf]["rawData"])
#     edfDict2[edf]["rawData"].plot_psd()

"""
fi = "sbs2data_2018_09_01_09_44_21_337.edf"
edfDict[fi]["rawData"].info
tensorFi = spectrogramMake(edfDict[fi]["rawData"], t0=edfDict[fi]["rawData"].first_samp, tWindow=edfDict[fi]["tWindow"])

test = slidingWindow(edfDict[fi], tN=edfDict[fi]["rawData"].last_samp,
                     tStep=edfDict[fi]["tStep"]*edfDict[fi]["fS"],
                     localSave={"sliceSave":True, "saveDir":saveDir+"fooTemp"})
"""

# debug mode
# edfDefDict.clear()
# edfDict.clear()
# edfDict2.clear()


############ TODO: proof of concepts'
