import os, mne, torch, re
from collections import defaultdict
from datetime import datetime
from mne.io import read_raw_edf
import numpy as np
# from ../LoadFarrahTueData.loadData import jsonLoad
import loadData
# import NovelEEG.SampleCode.loadData
from scipy import signal
import matplotlib.pyplot as plt

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\"  # ~~~ What is your execute path?
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"  # ~~~ What is the name of your data folder?
tuhData = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\01_tcp_ar\002"+"\\"
tuhData = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\**\01_tcp_ar"+"\\" #\100\00010023\s002_2013_02_21
# jsonDir = r"edfFiles.json" # ~~~ Where is your json folder?
jsonDir = r"tmp.json"
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

# PUT INTO TUH CALL after pipeline call for A TUH file
def TUHfooDef(EEGseries=False):
    david = EEGseries
    for i in david.info["ch_names"]:
        reSTR = r"(?<=EEG )(.*)(?=-REF)"
        reLowC = ['FP1', 'FP2', 'FZ', 'CZ', 'PZ']
        if re.search(reSTR, i) and re.search(reSTR, i).group() in reLowC:
            lowC = i[0:5]+i[5].lower()+i[6:]
            mne.channels.rename_channels(david.info, {i: re.findall(reSTR, lowC)[0]})
        elif re.search(reSTR, i):
            mne.channels.rename_channels(david.info, {i: re.findall(reSTR, i)[0]})
        else:
            print(i)
    print(david.info["ch_names"])
    return david
# 'ROC', 'LOC', 'EKG1', 'T1', 'T2', 'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR'

def readRawEdf(edfDict=None, saveDir='', tWindow=120, tStep=30,
               read_raw_edf_param={'preload':True, "stim_channel":"auto"}):
    edfDict["rawData"] = read_raw_edf(saveDir+edfDict["path"][-1], **read_raw_edf_param)
    tStart = edfDict["rawData"].annotations.orig_time - 60*60
    tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
    # with open(saveDir + edfDict["path"][-1], encoding="latin-1") as fp:
    #     first_line = fp.readline()
    # edfDict["age"] = re.search(r"(Age:\d+)", first_line).group()
    edfDict["t0"] = datetime.fromtimestamp(tStart)
    edfDict["tN"] = datetime.fromtimestamp(tStart + tLast)
    edfDict["tWindow"] = tWindow
    edfDict["tStep"] = tStep
    edfDict["fS"] = edfDict["rawData"].info["sfreq"]
    return edfDict

# pre-processing pipeline single file
def pipeline(EEGseries=None, lpfq=1, hpfq=40, notchfq=50, downSam=100, type="easycap-M1"):
    # Follows Makoto's_Preprocessing_Pipeline recommended for EEGlab's ICA
    # combined with Early State Preprocessing: PREP
    # Step 1 & 2: Check paths and import data -> is completed in readRawEdf
    # Step 3: Downsample -> is omitted
    # TODO: add downsampling to 100Hz
    #
    # To inspect the raw signal make the following calls as required
    #  EEGseries.plot()
    #  EEGseries.plot_psd()
    #  EEGseries.plot_sensors(show_names=True)
    #
    # Step 4: HP-filter [1Hz] -> BP-filter [1Hz; 40Hz] for this study
    EEGseries.filter(lpfq, hpfq, fir_design='firwin')
    # Step 5: Import channel info -> configure cap setup and channel names
    EEGseries.set_montage(mne.channels.make_standard_montage(kind=type, head_size=0.095), raise_if_subset=False)
    # Step 6 utilizing data knowledge for line-noise removal
    EEGseries.notch_filter(freqs=notchfq, notch_widths=5)
    EEGseries.resample(sfreq=downSam)
    # Step 7 inference statistics to annotate bad channels
    # TODO: "BADS" code from MNEi
    # TUTORIAL: https://mne.tools/stable/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html?highlight=interpolate_bads
    # Step 7.1 TUH (write detailed)
    # Step 8
    # EEGseries.interpolate_bads(reset_bads=True, origin='auto')
    # Step 9 Re-reference the data to average
    EEGseries.set_eeg_reference()
    # Re-reference math proff
    # https://sccn.ucsd.edu/wiki/Makoto's_preprocessing_pipeline#Why_should_we_add_zero-filled_channel_before_average_referencing.3F_.2808.2F09.2F2020_Updated.3B_prayer_for_Nagasaki.29
    # Step 10 through 15 is conducted in separate ICLabel or ANN
    return EEGseries

def spectrogramMake(EEGseries=None, t0=0, tWindow=120, cropFq=45):
    edfFs = EEGseries.info["sfreq"]
    chWindows = EEGseries.get_data(start=int(t0), stop=int(t0+tWindow), reject_by_annotation="omit", picks=['eeg'])
    fAx, tAx, Sxx = signal.spectrogram(chWindows, fs=edfFs)
    # fTemp, tTemp, SxxTemp = signal.spectrogram(chWindows[0], fs=edfFs)
    # plt.pcolormesh(tTemp, fTemp, np.log(SxxTemp))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title("channel spectrogram: "+EEGseries.ch_names[0])
    # plt.ylim(0,45)
    # plt.show()
    return torch.tensor(np.log(Sxx[:, fAx <= cropFq, :]+np.finfo(float).eps)) # for np del torch.tensor

def slidingWindow(edfInfo=None, tN=0, tStep=60, localSave={"sliceSave":False, "saveDir":os.getcwd()}):
    windowEEG = defaultdict(list)
    sampleWindow = edfInfo["tWindow"]*edfInfo["fS"]
    for i in range(0, tN, int(tStep)):
        windowKey = "window_%i_%i" % (i, i+sampleWindow)
        windowEEG[windowKey] = spectrogramMake(edfInfo["rawData"], t0=i, tWindow=sampleWindow)
    if (1+tN) % int(tStep) != 0:
        windowKey = "window_%i_%i" % (int(tN-sampleWindow), int(tN))
        windowEEG[windowKey] = spectrogramMake(edfInfo["rawData"], t0=int(tN-sampleWindow), tWindow=sampleWindow)
    if localSave["sliceSave"]:
        idDir = edfInfo["rawData"].filenames[0].split('\\')[-1].split('.')[0]
        if not os.path.exists(localSave["saveDir"] + "tempData\\"):
            os.mkdir(localSave["saveDir"] + "tempData\\")
        if not os.path.exists(localSave["saveDir"] + "tempData\\" + idDir):
            os.mkdir(localSave["saveDir"] + "tempData\\" + idDir)
        for k,v in windowEEG.items():
            torch.save(v, localSave["saveDir"] + "tempData\\%s\\%s.pt" % (idDir, k)) # for np del torch.tensor
    if not localSave["sliceSave"]:
        windowOut = windowEEG.copy()
    else:
        windowOut = None
    return windowOut

def completePrep(tWin=120, tStep=30, localSave={"sliceSave":False, "saveDir":os.getcwd()},
                 notchFQ=50, lpFQ=1, hpFQ=40):
    for edf in edfDict.keys():
        edfDict[edf] = readRawEdf(edfDict[edf], tWindow=tWin, tStep=tStep)
        pipeline(edfDict[edf]["rawData"], lpfq=lpFQ, hpfq=hpFQ, notchfq=notchFQ)
        tLastN = edfDict[fi]["rawData"].last_samp
        windowDict = slidingWindow(edfInfo=edfDict[edf], tN=tLastN, tStep=tStep, localSave=localSave)
        edfDict[edf]["tensors"] = windowDict
    return edfDict

# def annoLoad(edfFile=False, time=False, type=TUH):

# print("here")
# edfTUH = findEdf(tuhData, saveDir=saveDir)
# for edf in edfTUH:
#     edfTUH[edf] = readRawEdf(edfTUH[edf], saveDir=saveDir)
# for edf in edfTUH:
#     pipeline(edfTUH[edf]["rawData"])
# # load json to dict
edfDefDict = loadData.jsonLoad(path=jsonDataDir)

# load ALL .edf paths
# edfDict = edfDefDict.copy()
# for edf in edfDict:
#     edfDict[edf] = readRawEdf(edfDict[edf])
#     preprocessPipeline.pipeline(edfDict[edf]["rawData"])
#     # edfDict[edf]["rawData"].plot_psd()
edfBCDict = loadData.jsonLoad(path=saveDir+'nCompEdf.json')
BCfiles = ["sbs2data_2018_09_07_15_58_37_454.edf",
           "sbs2data_2018_09_07_11_05_24_439.edf"]
edfBC = defaultdict(dict)
for edf in BCfiles:
    edfBC[edf] = edfBCDict[edf].copy()
    edfBC[edf] = readRawEdf(edfBC[edf], saveDir=saveDir)
    pipeline(edfBC[edf]["rawData"])
    edfBC[edf]["rawData"].plot_psd()

edfBC[BCfiles[0]]["rawData"].plot_sensors(show_names=True)
# loading several files by [list]
fileSeveral = ["00010418_s008_t000.edf",
             "00010079_s004_t002.edf",
             "00009630_s001_t001.edf"]
fileSeveral = ["00009630_s001_t001.edf"]
# fileSeveral = ["sbs2data_2018_09_07_16_41_25_457.edf"]
edfDict2 = defaultdict(dict)
for edf in fileSeveral:
    edfDict2[edf] = edfDefDict[edf].copy()
    edfDict2[edf] = readRawEdf(edfDict2[edf], saveDir=saveDir,
                               read_raw_edf_param={'preload': True,
                                                   "stim_channel": ['EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF',
                                                                    'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF', 'IBI',
                                                                    'BURSTS', 'SUPPR']})
    edfDict2[edf]['annoDF'] = loadData.annoTUH(annoPath=edfDict2[edf]['path'][-1].split(".edf")[0] + ".tse",
                                              window=[0, 50000],
                                              saveDir=saveDir)
    edfDict2[edf]["rawData"] = TUHfooDef(edfDict2[edf]["rawData"])
    pipeline(edfDict2[edf]["rawData"], type="standard_1005", notchfq=60)
    slidingWindow(edfDict2[edf], tN=edfDict2[edf]["rawData"].last_samp,
                  tStep=edfDict2[edf]["tStep"] * edfDict2[edf]["fS"],
                  localSave={"sliceSave": True, "saveDir": saveDir})

edfDict2["00009630_s001_t001.edf"]["rawData"].plot_sensors(show_names=True)


    # edfDict[edf]["rawData"].plot_psd()
# type(edfDict["sbs2data_2018_09_01_09_08_16_335.edf"]["t0"])
edfDict2["00009630_s001_t001.edf"]["annoDF"].to_numpy().tolist()
for i in edfDict2:
    # edfDict2[i].pop("rawData", None)
    # edfDict2[i].pop("t0", None)
    # edfDict2[i].pop("tN", None)
    edfDict2[i]["annoDF"] = edfDict2[i]["annoDF"].to_numpy().tolist()


edfDict2["00009630_s001_t001.edf"]["annoDF"]

loadData.jsonSave("edin.json",saveDir,edfDict2)
# same method for loading single file
# fileSingle = ["sbs2data_2018_09_07_16_41_25_457.edf"]
# edfDict2 = defaultdict(dict)
# for edf in fileSingle:
#     edfDict2[edf] = edfDefDict[edf].copy()
#     edfDict2[edf] = readRawEdf(edfDict2[edf], saveDir=saveDir)
#     pipeline(edfDict2[edf]["rawData"])
#     edfDict2[edf]["rawData"].plot_psd()


fi = "sbs2data_2018_09_01_09_44_21_337.edf"
fi = fileSeveral[0]
edfDict2[fi]["rawData"].info
tensorFi = spectrogramMake(edfDict2[fi]["rawData"], t0=edfDict2[fi]["rawData"].first_samp, tWindow=edfDict2[fi]["tWindow"])

test = slidingWindow(edfDict2[fi], tN=edfDict2[fi]["rawData"].last_samp,
                     tStep=edfDict2[fi]["tStep"]*edfDict2[fi]["fS"],
                     localSave={"sliceSave":True, "saveDir":saveDir+"fooTemp"})



############ TODO: proof of concepts'

