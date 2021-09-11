import os, mne, torch, re, time
from collections import defaultdict
import numpy as np
import pandas as pd
# from ../LoadFarrahTueData.loadData import jsonLoad
import loadData
from preprocessPipeline import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow
from scipy import signal
import matplotlib.pyplot as plt

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
save_dir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\"  # ~~~ What is your execute path?
TUAR_dir_single_subject = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\01_tcp_ar\002"+"\\"
# TUAR_dir = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset"+"\\"
TUAR_dir = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\**\01_tcp_ar"+"\\" #\**\01_tcp_ar #\100\00010023\s002_2013_02_21
# TUAR_dir = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\**\02_tcp_le"+"\\" # debug 02_tcp_le
# TUAR_dir = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\**\03_tcp_ar_a"+"\\" # debug 03_tcp_ar_a
# jsonDir = r"edfFiles.json" # ~~~ Where is your json folder?
jsonDir = r"tmp.json"
jsonDataDir = save_dir + jsonDir
TUAR_dirDir = save_dir + TUAR_dir

TUAR_data = loadData.findEdf(path=TUAR_dir, selectOpt=False, saveDir=save_dir)
tutorial_single = ["00009630_s001_t001.edf"]
tutorial_prep = ["00010418_s008_t000.edf", "00010079_s004_t002.edf", "00009630_s001_t001.edf", '00007952_s001_t001.edf']
tutorial_model = ["00010418_s008_t000.edf", "00010079_s004_t002.edf", "00009630_s001_t001.edf", '00007952_s001_t001.edf',
               '00009623_s008_t004.edf', '00009623_s008_t005.edf', '00009623_s010_t000.edf',
               '00001006_s001_t001.edf', '00006501_s001_t000.edf', '00006514_s008_t001.edf', '00006514_s020_t001.edf']

# for all subjects run as: file_selected = TUAR_data
file_selected = TUAR_data.copy()

# prepare TUAR output
counter = 0 # debug counter
tic = time.time()

subjects = defaultdict(dict)
all_subject_gender = {"male": [], "female": [], "other": []}
all_subject_age = []
for edf in file_selected: #TUAR_data:
    subject_ID = edf.split('_')[0]
    if subject_ID in subjects.keys():
        subjects[subject_ID][edf] = TUAR_data[edf].copy()
    else:
        subjects[subject_ID] = {edf: TUAR_data[edf].copy()}

    # debug counter for subject error
    counter += 1
    print("\n%s is patient: %i" % (edf, counter))

    # initialize hierarchical dict
    proc_subject = subjects[subject_ID][edf]
    proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1*.25,
                              read_raw_edf_param={'preload': True}) #,
                                                  # "stim_channel": ['EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF',
                                                  #                  'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF', 'IBI',
                                                  #                  'BURSTS', 'SUPPR']})

    # find data labels
    labelPath = subjects[subject_ID][edf]['path'][-1].split(".edf")[0]
    proc_subject['annoDF'] = loadData.label_TUH_full(annoPath=labelPath+".tse", window=[0, 50000], saveDir=save_dir)

    # Makoto + PREP processing steps
    proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
    # ch_TPC = mne.pick_channels(proc_subject["rawData"].info['ch_names'],
    #                            include=['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz',
    #                                     'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6'],
    #                           exclude=['Fz', 'Pz', 'ROC', 'LOC', 'EKG1', 'T1', 'T2', 'BURSTS', 'SUPPR', 'IBI', 'PHOTIC'])
    #                           exclude2 = [, 'A1', 'A2']
    # mne.pick_info(proc_subject["rawData"].info, sel=ch_TPC, copy=False)
    # proc_subject["rawData"].pick_channels(proc_subject["rawData"].info, sel=ch_TPC, copy=False)
    proc_subject["rawData"].pick_channels(ch_names=['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz',
                                                    'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2'])
    pipeline(proc_subject["rawData"], type="standard_1005", notchfq=60, downSam=150)

    # Generate output windows for (X,y) as (tensor, label)
    # proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times.max(),
    #                                                      tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
    #                                                      annoDir=save_dir,
    #              localSave={"sliceSave": False, "saveDir": save_dir, "local_return": False}) #r"C:\Users\anden\PycharmProjects"+"\\"})

    # catch age and gender for descriptive statistics
    if subjects[subject_ID][edf]["gender"].lower() == 'm':
        all_subject_gender["male"].append(subjects[subject_ID][edf]["gender"].lower())
        # gender[0].append(subjects[id][edf]["gender"].lower())
    elif subjects[subject_ID][edf]["gender"].lower() == 'f':
        all_subject_gender["female"].append(subjects[subject_ID][edf]["gender"].lower())
        # gender[1].append(subjects[id][edf]["gender"].lower())
    else:
        all_subject_gender["other"].append(subjects[subject_ID][edf]["gender"].lower())
        # print(subjects[id][edf]["gender"].lower())
    all_subject_age.append(subjects[subject_ID][edf]["age"])
    # except:
    #     print("sit a while and listen: %s" % subjects[subject_ID][edf]['path'])

all_subject_age = np.array(all_subject_age)

toc = time.time()
print("\n~~~~~~~~~~~~~~~~~~~~\n"
      "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
      "\n~~~~~~~~~~~~~~~~~~~~\n"
      % (int((toc-tic)/60), int((toc-tic) % 60), len(subjects), subjects[subject_ID][edf]["tWindow"], subjects[subject_ID][edf]["tStep"]))

# result inspection
pID = -1
p_inspect = list(file_selected)[pID]
subjects[p_inspect.split('_')[0]][p_inspect]["rawData"].plot_sensors(show_names=True) # view electrode placement
# all_ch = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2']
subjects[p_inspect.split('_')[0]][p_inspect]["rawData"].plot(remove_dc=True) # plot data as electrodes-amp/samples
subjects[p_inspect.split('_')[0]][p_inspect]["annoDF"] # show annotation sections
subject_prep_output = list(subjects[p_inspect.split('_')[0]][p_inspect]["preprocessing_output"].values()) # segmented windows for models

all_subject_age_hist = np.histogram(all_subject_age, range=(0,100))
plt.hist(all_subject_age, range=(0,100))
plt.show()

print("end of script - fagprojekt tutorial")
