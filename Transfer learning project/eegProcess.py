import os, mne, torch, re, warnings
from collections import defaultdict
from datetime import datetime, timezone
from mne.io import read_raw_edf
import numpy as np
# from ../LoadFarrahTueData.eegLoader import jsonLoad
import eegLoader
# import NovelEEG.SampleCode.eegLoader
from scipy import signal, stats
import matplotlib.pyplot as plt


def readRawEdf(edfDict=None, saveDir='', tWindow=120, tStep=30,
               read_raw_edf_param={'preload': True, "stim_channel": "auto"}):
    try:
        edfDict["rawData"] = read_raw_edf(saveDir+edfDict["path"][-1], **read_raw_edf_param)
        edfDict["fS"] = edfDict["rawData"].info["sfreq"]
        t_start = edfDict["rawData"].annotations.orig_time
        if t_start.timestamp() <= 0:
            edfDict["t0"] = datetime.fromtimestamp(0, tz=timezone.utc)
            t_last = edfDict["t0"].timestamp() + edfDict["rawData"]._last_time + 1 / edfDict["fS"]
            edfDict["tN"] = datetime.fromtimestamp(t_last, tz=timezone.utc)
        else:
            t_last = t_start.timestamp() + edfDict["rawData"]._last_time + 1 / edfDict["fS"]
            edfDict["t0"] = t_start  # datetime.fromtimestamp(t_start.timestamp(), tz=timezone.utc)
            edfDict["tN"] = datetime.fromtimestamp(t_last, tz=timezone.utc)

        edfDict["tWindow"] = float(tWindow) # width of EEG sample window, given in (sec)
        edfDict["tStep"] = float(tStep) # step/overlap between EEG sample windows, given in (sec)

        # header = mne.io.edf.edf._read_edf_header(fname=edfDict["rawData"].filenames[0], exclude=[])
        # id_tmp = header[0]["subject_info"]["id"]
        # with open(saveDir + edfDict["path"][-1], encoding="latin-1") as fp:
        #     first_line = fp.readline()
        #
        # # regex_found = re.findall(r"%s \S+" % id_tmp, first_line)
        # subject_gender = re.search(r"%s \w+" % id_tmp, first_line).group().split(" ")[1]
        # subject_age = re.search(r"%s (Age:\d+)" % id_tmp, first_line).group().split(":")[1]
        # edfDict["gender"] = subject_gender
        # edfDict["age"] = int(subject_age)
    except:
        print("error break please inspect:\n %s\n~~~~~~~~~~~~" % edfDict["rawData"].filenames[0])

    return edfDict

# pre-processing pipeline single file
def pipeline(MNE_raw=None, lpfq=1, hpfq=40, notchfq=50, downSam=100, cap_setup="easycap-M1"):
    # Follows Makoto's_Preprocessing_Pipeline recommended for EEGlab's ICA
    # combined with Early State Preprocessing: PREP
    # Step 1 & 2: Check paths and import data -> is completed in readRawEdf
    #
    # To inspect the raw signal make the following calls as required
    #  MNE_raw.plot()
    #  MNE_raw.plot_psd()
    #  MNE_raw.plot_sensors(show_names=True)
    #
    # Step 4: HP-filter [1Hz] -> BP-filter [1Hz; 40Hz] for this study
    # Roy BP(0.1, 70)?, our BP(1, 40)
    MNE_raw.filter(lpfq, hpfq, fir_design='firwin')
    # Step 5: Import channel info -> configure cap setup and channel names
    # Roy TCP-montage 22-channels, our mean-average-montage
    MNE_raw.set_montage(mne.channels.make_standard_montage(kind=cap_setup, head_size=0.095), on_missing="warn")
    # Step 6 utilizing data knowledge for line-noise removal
    # Roy none?, our notch(cut=60, width=5)
    MNE_raw.notch_filter(freqs=notchfq, notch_widths=5)

    # Step : Downsample
    MNE_raw.resample(sfreq=downSam)
    # Roy 200, our 250

    # Step 7 inference statistics to annotate bad channels
    # TODO: "BADS" code from MNEi & PREP
    # TUTORIAL: https://mne.tools/stable/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html?highlight=interpolate_bads
    # Step 7.1 TUH (write detailed)
    # Step 8
    # MNE_raw.interpolate_bads(reset_bads=True, origin='auto')
    # Step 9 Re-reference the data to average
    MNE_raw.set_eeg_reference()
    # Re-reference math proff
    # https://sccn.ucsd.edu/wiki/Makoto's_preprocessing_pipeline#Why_should_we_add_zero-filled_channel_before_average_referencing.3F_.2808.2F09.2F2020_Updated.3B_prayer_for_Nagasaki.29
    # Step 10 through 15 is conducted in separate ICLabel or ANN

    return MNE_raw


def nonPipeline(MNE_raw=None, hp_cut=1, down_sam=250, cap_setup="easycap-M1"):
    # IDEA -> mne.hp.montage.downsamle.reref
    # To inspect the raw signal make the following calls as required
    # MNE_raw.plot()
    # MNE_raw.plot_psd()
    # MNE_raw.plot_sensors(show_names=True)
    #
    # Import channel info -> configure cap setup and channel names
    # Roy TCP-montage 22-channels

    # apply high-pass
    MNE_raw.filter(l_freq=hp_cut, h_freq=None, fir_design='firwin')
    # load channels to 10_05 standard system
    MNE_raw.set_montage(mne.channels.make_standard_montage(kind=cap_setup, head_size=0.095), on_missing="warn")
    # downsample to equal 250hz
    MNE_raw.resample(sfreq=down_sam)
    # set common reference
    MNE_raw.set_eeg_reference()

    return MNE_raw


# calculate STFT, FFT or Spectrogram of EEG channels
def spectrogramMake(MNE_raw=None, t0=0, tWindow=120, crop_fq=45, FFToverlap=None, show_chan_num=None):
    try:
        edfFs = MNE_raw.info["sfreq"]
        chWindows = MNE_raw.get_data(start=int(t0), stop=int(t0+tWindow), reject_by_annotation="omit", picks=['eeg'])

        if FFToverlap is None:
            specOption = {"x": chWindows, "fs": edfFs, "mode": "psd"}
        else:
            window = signal.get_window(window=('tukey', 0.25), Nx=int(tWindow))  # TODO: error in 'Nx' & 'noverlap' proportions
            specOption = {"x": chWindows, "fs": edfFs, "window": window, "noverlap": int(tWindow*FFToverlap), "mode": "psd"}

        fAx, tAx, Sxx = signal.spectrogram(**specOption)
        normSxx = stats.zscore(np.log(Sxx[:, fAx <= crop_fq, :] + np.finfo(float).eps))
        if isinstance(show_chan_num, int):
            plot_spec = plotSpec(ch_names=MNE_raw.info['ch_names'], chan=show_chan_num,
                                 fAx=fAx[fAx <= crop_fq], tAx=tAx, Sxx=normSxx)
            plot_spec.show()
    except:
        print("pause here")
        # fTemp, tTemp, SxxTemp = signal.spectrogram(chWindows[0], fs=edfFs)
        # plt.pcolormesh(tTemp, fTemp, np.log(SxxTemp))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.title("channel spectrogram: "+MNE_raw.ch_names[0])
        # plt.ylim(0,45)
        # plt.show()

    return torch.tensor(normSxx.astype(np.float16)) # for np delete torch.tensor
# TODO benchmark feature
# foo = normSxx.reshape([19,241]) full FFT (cutted)
# eig_foo = np.linalg.eig(np.cov(foo))
# feat_extract_bench = eig_foo[1]

# Segment EEG into designated windows
def slidingWindow(EEG_series=None, t_max=0, tStep=1, FFToverlap=None, crop_fq=45, annoDir=None,
                  localSave={"sliceSave":False, "saveDir":os.getcwd(), "local_return":False}):
    # catch correct sample frequency and end sample
    edf_fS = EEG_series["rawData"].info["sfreq"]
    t_N = int(t_max*edf_fS)

    # ensure window-overlaps progress in sample interger
    if float(tStep*edf_fS) == float(int(tStep*edf_fS)):
        t_overlap = int(tStep*edf_fS)
    else:
        t_overlap = int(tStep*edf_fS)
        overlap_change = 100-(t_overlap/edf_fS)*100
        print("\n  tStep [%.3f], overlap does not equal an interger [%f] and have been rounded to %i"
              "\n  equaling to %.1f%% overlap or %.3fs time steps\n\n"
              % (tStep, tStep*edf_fS, t_overlap, overlap_change, t_overlap/edf_fS))

    # initialize variables for segments
    window_EEG = defaultdict(tuple)
    window_width = int(EEG_series["tWindow"]*edf_fS)
    label_path = EEG_series['path'][-1].split(".edf")[0] + ".tse"

    # segment all N-1 windows (by positive lookahead)
    for i in range(0, t_N-window_width, t_overlap):
        t_start = i/edf_fS
        t_end = (i+window_width)/edf_fS
        window_key = "window_%.3fs_%.3fs" % (t_start, t_end)
        window_data = spectrogramMake(EEG_series["rawData"], t0=i, tWindow=window_width,
                                      FFToverlap=FFToverlap, crop_fq=crop_fq) # , show_chan_num=0) #)
        window_label = eegLoader.label_TUH(annoPath=label_path, window=[t_start, t_end], saveDir=annoDir)
        window_EEG[window_key] = (window_data, window_label)
    # window_N segments (by negative lookahead)
    if t_N % t_overlap != 0:
        t_start = (t_N - window_width)/edf_fS
        t_end = t_N/edf_fS
        window_key = "window_%.3fs_%.3fs" % (t_start, t_end)
        window_data = spectrogramMake(EEG_series["rawData"], t0=t_start, tWindow=window_width,
                                                 FFToverlap=FFToverlap, crop_fq=crop_fq)
        window_label = eegLoader.label_TUH(annoPath=label_path, window=[t_start, t_end], saveDir=annoDir)
        window_EEG[window_key] = (window_data, window_label)

    # save in RAM, disk or not
    if localSave["sliceSave"]:
        idDir = EEG_series["rawData"].filenames[0].split('\\')[-1].split('.')[0]
        if not os.path.exists(localSave["saveDir"] + "tempData\\"):
            os.mkdir(localSave["saveDir"] + "tempData\\")
        if not os.path.exists(localSave["saveDir"] + "tempData\\" + idDir):
            os.mkdir(localSave["saveDir"] + "tempData\\" + idDir)
        for k, v in window_EEG.items():
            torch.save(v, localSave["saveDir"] + "tempData\\%s\\%s.pt" % (idDir, k)) # for np del torch.save
    if not localSave["sliceSave"] or localSave["local_return"] is True:
        windowOut = window_EEG.copy()
    else:
        windowOut = None

    return windowOut

# renames TUH channels to conventional 10-20 system
def TUH_rename_ch(MNE_raw=False):
    # MNE_raw
    # mne.channels.rename_channels(MNE_raw.info, {"PHOTIC-REF": "PROTIC"})
    for i in MNE_raw.info["ch_names"]:
        reSTR = r"(?<=EEG )(\S*)(?=-REF)"  # working reSTR = r"(?<=EEG )(.*)(?=-REF)"
        reLowC = ['FP1', 'FP2', 'FZ', 'CZ', 'PZ']

        if re.search(reSTR, i) and re.search(reSTR, i).group() in reLowC:
            lowC = i[0:5]+i[5].lower()+i[6:]
            mne.channels.rename_channels(MNE_raw.info, {i: re.findall(reSTR, lowC)[0]})
        elif i == "PHOTIC-REF":
            mne.channels.rename_channels(MNE_raw.info, {i: "PHOTIC"})
        elif re.search(reSTR, i):
            mne.channels.rename_channels(MNE_raw.info, {i: re.findall(reSTR, i)[0]})
        else:
            continue
            # print(i)
    print(MNE_raw.info["ch_names"])
    return MNE_raw

# plot function for spectrograms
def plotSpec(ch_names=False, chan=False, fAx=False, tAx=False, Sxx=False):
    # fTemp, tTemp, SxxTemp = signal.spectrogram(chWindows[0], fs=edfFs)
    # normSxx = stats.zscore(np.log(Sxx[:, fAx <= cropFq, :] + np.finfo(float).eps))
    plt.pcolormesh(tAx, fAx, Sxx[chan, :, :])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("channel spectrogram: " + ch_names[chan])

    return plt

# THIS FUNCTION IS CURRENTLY BROKEN!
# def completePrep(tWin=120, tStep=30, localSave={"sliceSave":False, "saveDir":os.getcwd()},
#                  notchFQ=50, lpFQ=1, hpFQ=40):
#     for edf in edfDict.keys():
#         edfDict[edf] = readRawEdf(edfDict[edf], tWindow=tWin, tStep=tStep)
#         pipeline(edfDict[edf]["rawData"], lpfq=lpFQ, hpfq=hpFQ, notchfq=notchFQ)
#         tLastN = edfDict[fi]["rawData"].last_samp
#         windowDict = slidingWindow(MNE_raw=edfDict[edf], tN=tLastN, tStep=tStep, localSave=localSave)
#         edfDict[edf]["tensors"] = windowDict
#
#     return edfDict
# THIS FUNCTION IS CURRENTLY BROKEN!

# ############ TODO:
#
