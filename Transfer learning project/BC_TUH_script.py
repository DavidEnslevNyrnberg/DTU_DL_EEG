# Input:
# .best_model_PARAMs > Y_true
#
# Modes:
#
#
# Output:
# Should return the full evaluation for all prediction:
# to.csv_(conf, report, Y_true.plot, Y_pred.plot, Y_FN.plot, Y_FP.plot)
#





import os, mne, torch, re, time
from collections import defaultdict
import numpy as np
import pandas as pd
# from ../LoadFarrahTueData.eegLoader import jsonLoad
import eegLoader
from eegProcess import TUH_rename_ch, readRawEdf, nonPipeline, spectrogramMake, slidingWindow, pipeline
from scipy import signal
import matplotlib.pyplot as plt

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
save_dir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\"  # ~~~ What is your execute path?
save_tensor = r'C:\Users\anden\Documents\PersonalProjects'+'\\'
BC_dir = r'data_farrahtue_EEG\custom_BC_select'+"\\"
TUAR_dir = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\**\01_tcp_ar"+"\\"
data_dir = save_dir+BC_dir
BC_data = eegLoader.findEdf(path=BC_dir, selectOpt=False, saveDir=save_dir)
TUAR_data = eegLoader.findEdf(path=TUAR_dir, selectOpt=False, saveDir=save_dir)

# BC IDs [349, 300, 350, 337]
tutorial_BC = ["sbs2data_2018_09_03_08_51_58_349.edf", "sbs2data_2018_08_31_01_49_59_300.edf",
               "sbs2data_2018_09_03_09_55_18_350.edf", "sbs2data_2018_09_01_09_44_21_337.edf"]
tutorial_TUAR = ["00009630_s001_t001.edf"]
quali = [10, 9, 1, 1]

file_selected = tutorial_BC.copy()
file_selected_TUAR = TUAR_data.copy()

# prepare TUAR output
counter = 0  # debug counter
tic = time.time()

# subjects = defaultdict(dict)
# # all_subject_gender = {"male": [], "female": [], "other": []}
# # all_subject_age = []
# for edf in file_selected:
#     subject_ID = edf.split('_')[0]
#     if subject_ID in subjects.keys():
#         subjects[subject_ID][edf] = BC_data[edf].copy()
#     else:
#         subjects[subject_ID] = {edf: BC_data[edf].copy()}
#
#     # debug counter for subject error
#     counter += 1
#     print("\n%s is patient: %i\n" % (edf, counter))
#
#     # initialize hierarchical dict
#     proc_subject = subjects[subject_ID][edf]
#     proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1*.25,
#                               read_raw_edf_param={'preload': True})
#
#     # find data labels
#     # labelPath = subjects[subject_ID][edf]['path'][-1].split(".edf")[0]
#     # proc_subject['annoDF'] = eegLoader.label_TUH_full(annoPath=labelPath+".tse", window=[0, 50000], saveDir=save_dir)
#
#     # prepare raw
#     # proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
#     TUH_pick = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz', 'Fp2', 'F4',
#                 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2']
#     BC_pick = ['TP10', 'Fz', 'P3', 'Cz', 'C4', 'TP9', 'P4', 'FT7', 'C3', 'O1', 'FT8', 'Fpz', 'O2', 'Pz']
#     proc_subject["rawData"].pick_channels(ch_names=BC_pick)
#     # 'easycap-M1' for 2d-projected electrodes
#     # 'standard_1005' for 3d electrodes
#     nonPipeline(proc_subject["rawData"], cap_setup="standard_1005", hp_cut=1, down_sam=250)
#
#     # Generate output windows for (X,y) as (tensor, label)
#     proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times[-1],
#                                                          tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
#                                                          annoDir=[],
#                                                          localSave={"sliceSave": False, "saveDir": save_tensor+r'../BC14', "local_return": True}) #r"C:\Users\anden\PycharmProjects"+"\\"})
#
# fooN = subjects['sbs2data']['sbs2data_2018_09_03_09_55_18_350.edf']['preprocessing_output']['window_22.072s_23.072s'][0].numpy().astype(np.float16)
# z = fooN[:, :, 0]
# x_, y_ = np.meshgrid(np.arange(0, 25, 1), np.arange(0, 14, 1))
# z_grid = np.array(z).reshape(14, 25)
# fig = plt.figure(1)
# ax1 = plt.contourf(x_, y_, z_grid)
# plt.show()
#
# extent = [0, 25, 0, 14]
# fig2 = plt.figure(2, figsize=(16, 12))
# ax2 = fig2.add_subplot(111)
# im = ax2.imshow(z, extent=extent, origin='lower', interpolation='None', cmap='viridis')
# plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
#
# fig2 = plt.figure(2)
# ax1 = fig2.add_subplot(121)
# #  Bilinear interpolation - this will look blurry
# ax1.imshow(z_grid, interpolation='bilinear', cmap='viridis')
#
# ax2 = fig2.add_subplot(122)
# # 'nearest' interpolation - faithful but blocky
# ax2.imshow(z_grid, interpolation='nearest', cmap='viridis')
#
# plt.show()
#
# subjects_TUAR = defaultdict(dict)
# # all_subject_gender = {"male": [], "female": [], "other": []}
# # all_subject_age = []
# for edf in file_selected_TUAR:
#     subject_ID = edf.split('_')[0]
#     if subject_ID in subjects_TUAR.keys():
#         subjects_TUAR[subject_ID][edf] = TUAR_data[edf].copy()
#     else:
#         subjects_TUAR[subject_ID] = {edf: TUAR_data[edf].copy()}
#
#     # debug counter for subject error
#     counter += 1
#     print("\n%s is patient: %i\n" % (edf, counter))
#     # if counter <= 291 or edf in TUAR_data:
#     #      continue
#
#     # initialize hierarchical dict
#     proc_subject = subjects_TUAR[subject_ID][edf]
#     proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1*.25,
#                               read_raw_edf_param={'preload': True})
#
#     # find data labels
#     # labelPath = subjects[subject_ID][edf]['path'][-1].split(".edf")[0]
#     # proc_subject['annoDF'] = eegLoader.label_TUH_full(annoPath=labelPath+".tse", window=[0, 50000], saveDir=save_dir)
#
#     # prepare raw
#     proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
#     TUH_pick = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4',
#                 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz']
#     TUH_to_BC_picks = {'T3': 'TP9', 'F7': 'FT7', 'Fp2': 'Fpz', 'F8': 'FT8', 'T4': 'TP10',
#                        'Fz': 'Fz', 'C3': 'C3', 'P3': 'P3', 'O1': 'O1', 'Cz': 'Cz', 'C4': 'C4', 'P4': 'P4', 'O2': 'O2'}
#     mne.channels.rename_channels(proc_subject["rawData"].info, mapping=TUH_to_BC_picks)
#     BC_pick = ['TP10', 'Fz', 'P3', 'Cz', 'C4', 'TP9', 'P4', 'FT7', 'C3', 'O1', 'FT8', 'Fpz', 'O2', 'Pz']
#     proc_subject["rawData"].pick_channels(ch_names=BC_pick)
#     proc_subject["rawData"].reorder_channels(BC_pick)
#     nonPipeline(proc_subject["rawData"], cap_setup="standard_1005", hp_cut=1, down_sam=250) # "standard_1005" "easycap-M1"
#
#     # Generate output windows for (X,y) as (tensor, label)
#     proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times[-1],
#                                                          tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
#                                                          annoDir=save_dir,
#                                                          localSave={"sliceSave": True, "saveDir": save_tensor+r'./TUAR14_2nd', "local_return": True}) #r"C:\Users\anden\PycharmProjects"+"\\"})

# fooM = subjects_TUAR['00009630']['00009630_s001_t001.edf']['preprocessing_output']['window_22.072s_23.072s'][0].numpy().astype(np.float16)
# z = fooM[:, :, 0]
# x_, y_ = np.meshgrid(np.arange(0,25,1), np.arange(0,14,1))
# z_grid = np.array(z).reshape(14, 25)
# fig = plt.figure(2)
# ax2 = plt.contour(x_,y_,z_grid)
# plt.show()


subjects_TUAR19 = defaultdict(dict)
# all_subject_gender = {"male": [], "female": [], "other": []}
# all_subject_age = []
for edf in file_selected_TUAR:
    subject_ID = edf.split('_')[0]
    if subject_ID in subjects_TUAR19.keys():
        subjects_TUAR19[subject_ID][edf] = TUAR_data[edf].copy()
    else:
        subjects_TUAR19[subject_ID] = {edf: TUAR_data[edf].copy()}

    # debug counter for subject error
    counter += 1
    print("\n%s is patient: %i\n" % (edf, counter))

    # initialize hierarchical dict
    proc_subject = subjects_TUAR19[subject_ID][edf]
    proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1*.25,
                              read_raw_edf_param={'preload': True})

    # find data labels
    # labelPath = subjects[subject_ID][edf]['path'][-1].split(".edf")[0]
    # proc_subject['annoDF'] = eegLoader.label_TUH_full(annoPath=labelPath+".tse", window=[0, 50000], saveDir=save_dir)

    # prepare raw
    proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
    TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Cz']
    # TUH_to_BC_picks = {'T3': 'TP9', 'F7': 'FT7', 'Fp2': 'Fpz', 'F8': 'FT8', 'T4': 'TP10',
    #                    'Fz': 'Fz', 'Cz': 'Cz', 'Pz': 'Pz',
    #                    'C3': 'C3', 'P3': 'P3', 'O1': 'O1', 'C4': 'C4', 'P4': 'P4', 'O2': 'O2'}
    # mne.channels.rename_channels(proc_subject["rawData"].info, mapping=TUH_to_BC_picks)
    # BC_pick = ['TP10', 'Fz', 'P3', 'Cz', 'C4', 'TP9', 'P4', 'FT7', 'C3', 'O1', 'FT8', 'Fpz', 'O2', 'Pz']
    proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
    proc_subject["rawData"].reorder_channels(TUH_pick)
    pipeline(proc_subject["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=40, notchfq=60, downSam=250) # "standard_1005" "easycap-M1"

    # Generate output windows for (X,y) as (tensor, label)
    proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times[-1],
                                                         tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
                                                         annoDir=save_dir,
                                                         localSave={"sliceSave": True, "saveDir": save_tensor+r'/TUAR19_new_24hz', "local_return": False}) #r"C:\Users\anden\PycharmProjects"+"\\"})

toc = time.time()
print("\n~~~~~~~~~~~~~~~~~~~~\n"
      "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
      "\n~~~~~~~~~~~~~~~~~~~~\n"
      % (int((toc-tic)/60), int((toc-tic) % 60), len(subjects_TUAR19),
         subjects_TUAR19[subject_ID][edf]["tWindow"], subjects_TUAR19[subject_ID][edf]["tStep"]))

# result inspection
pID = -1
p_inspect = list(file_selected)[pID]
# subjects[p_inspect.split('_')[0]][p_inspect]["rawData"].plot_sensors(show_names=True) # view electrode placement
# all_ch = ['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2']
# subjects[p_inspect.split('_')[0]][p_inspect]["rawData"].plot(remove_dc=True) # plot data as electrodes-amp/samples
# subjects[p_inspect.split('_')[0]][p_inspect]["annoDF"] # show annotation sections
# subject_prep_output = list(subjects[p_inspect.split('_')[0]][p_inspect]["preprocessing_output"].values()) # segmented windows for models

# all_subject_age_hist = np.histogram(all_subject_age, range=(0, 100))
# plt.hist(all_subject_age, range=(0, 100))
# plt.show()

foo = subjects[p_inspect.split('_')[0]][p_inspect]["annoDF"]
foo["n_start"] = (foo["t_start"]*250).astype(int)
foo["n_end"] = (foo["t_end"]*250).astype(int)


counter_total = {"id": list(), "sess": [], "file": [], "sec": []}
counter_label = {"eyem": {"id": list(), "sess": [], "file": [], "sec": []},
                 "chew": {"id": list(), "sess": [], "file": [], "sec": []},
                 "shiv": {"id": list(), "sess": [], "file": [], "sec": []},
                 "elpp": {"id": list(), "sess": [], "file": [], "sec": []},
                 "musc": {"id": list(), "sess": [], "file": [], "sec": []},
                 "null": {"id": list(), "sess": [], "file": [], "sec": []}}
imageDict = defaultdict()

foo = []
for k in subjects:
    # counter_total["id"].append(k)
    for v in subjects[k]:
        foo.append(subjects[k][v]["rawData"].ch_names)
        # counter_total["sess"].append("_".join(v.split("_")[0:2]))
        # counter_total["file"].append(v)
        # counter_total["sec"].append(subjects[k][v]["tN"].timestamp()-subjects[k][v]["t0"].timestamp())
        # for lab in counter_label.keys():
        #     if subjects[k][v]["annoDF"]["label"].str.contains(lab).any():
        #         counter_label[lab]["id"].append(k)
        #         counter_label[lab]["sess"].append("_".join(v.split("_")[0:2]))
        #         counter_label[lab]["file"].append(v)
        #         pd_label = subjects[k][v]["annoDF"].loc[subjects[k][v]["annoDF"]["label"] == lab]
        #         counter_label[lab]["sec"].append(pd_label[["t_start", "t_end"]].diff(axis=1).sum(axis=0)[1])


# all table
print([len(counter["id"]), len(set(counter["sess"])),
       len(counter["file"]), np.around(np.array(counter["sec"]).sum(), 2)])
# eyem table
print([len(set(counter_label["eyem"]["id"])), len(set(counter_label["eyem"]["sess"])),
       len(set(counter_label["eyem"]["file"])), np.around(np.array(counter_label["eyem"]["sec"]).sum(), 2)])
# chew table
print([len(set(counter_label["chew"]["id"])), len(set(counter_label["chew"]["sess"])),
       len(set(counter_label["chew"]["file"])), np.around(np.array(counter_label["chew"]["sec"]).sum(), 2)])
# shiv table
print([len(set(counter_label["shiv"]["id"])), len(set(counter_label["shiv"]["sess"])),
       len(set(counter_label["shiv"]["file"])), np.around(np.array(counter_label["shiv"]["sec"]).sum(), 2)])
# elpp table
print([len(set(counter_label["elpp"]["id"])), len(set(counter_label["elpp"]["sess"])),
       len(set(counter_label["elpp"]["file"])), np.around(np.array(counter_label["elpp"]["sec"]).sum(), 2)])
# musc table
print([len(set(counter_label["musc"]["id"])), len(set(counter_label["musc"]["sess"])),
       len(set(counter_label["musc"]["file"])), np.around(np.array(counter_label["musc"]["sec"]).sum(), 2)])
# null table
print([len(set(counter_label["null"]["id"])), len(set(counter_label["null"]["sess"])),
       len(set(counter_label["null"]["file"])), np.around(np.array(counter_label["null"]["sec"]).sum(), 2)])

print("wait here")
