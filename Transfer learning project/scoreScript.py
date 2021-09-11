# score Script
# This script/function/class should:
# load two numpy BC and TUAR NON-edited corpora
# --- TUAR yep
# load the pickle model trained on TUAR
# #EASY repeat tables from thesis >pandasFun.py<
# #NEW-?? make blue plots of:
# --- TUAR_test14 (target) [full]
# --- TUAR_test14 (pred) [full]
# --- TUAR_test14 diff(target,pred)
# --- BC all
# --- BC low-quali vs high-quali
# >pickle< should be used to safe/load the model across scripts for scoring
# DAVID NOTE:

# system imports
import os, time, random, pickle, glob, re, platform, itertools
# EEG imports
import mne, torch, hpsklearn
import eegLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from collections import defaultdict
from hpsklearn import HyperoptEstimator, linear_discriminant_analysis, one_vs_rest
from hyperopt import hp
from sklearn.metrics import f1_score, recall_score, classification_report, balanced_accuracy_score, accuracy_score


def score_model(model, Y_target, X_pred, classes, mod_name='foo'):
    model_predict = model.predict(X_pred)
    report = classification_report(Y_target[:, classes[1]], model_predict, target_names=classes[0],
                                   zero_division=0, output_dict=True)
    name = mod_name
    wF1 = report["weighted avg"]["f1-score"]
    acc = 1 - model.score(X_test, Y_target[:, classes[1]])
    bAcc = report["macro avg"]["recall"]
    sens = recall_score(Y_target[:, classes[1]], model_predict, average='weighted')
    s_eyem = report['eyem']["recall"]
    s_musc = report['musc']["recall"]
    s_null = report['null']["recall"]
    acc_eyem = accuracy_score(Y_target[:, 0], model_predict[:, 0])
    acc_musc = accuracy_score(Y_target[:, 4], model_predict[:, -2])
    acc_null = accuracy_score(Y_target[:, 5], model_predict[:, -1])
    score_list = [name, wF1, acc, bAcc, sens, s_eyem, s_musc, s_null, acc_eyem, acc_musc, acc_null]
    score_list[1:] = np.array(score_list[1:]).round(3)
    return score_list


def blue_plot(img, y_name, x_name, names, temporal, **kwargs):
    # def plot_something(data, ax=None, **kwargs):
    #     ax = ax or plt.gca()
    #     # Do some cool data transformations...
    #     return ax.boxplot(data, **kwargs)
    # fig, (ax1, ax2) = plt.subplots(2)
    # plot_something(data1, ax1, color='blue')
    # plot_something(data2, ax2, color='red')

    title_name = names[0]
    class_name = names[1]

    fixed_image = np.ones((len(img), temporal))*-1
    for i in range(len(img)):
        values = img[i]
        n = min(temporal, len(values))
        fixed_image[i, :n] = values[:n]

    plt.figure(figsize=(30, 3))
    t_size = 16
    plt.imshow(fixed_image, **kwargs)
    plt.xlabel(x_name, fontsize=t_size)
    plt.xticks(fontsize=t_size)
    plt.ylabel(y_name, fontsize=t_size)
    plt.yticks(fontsize=t_size)
    plt.title(title_name, fontsize=t_size)

    cbar = plt.colorbar()
    if isinstance(class_name, list):
        cbar.set_ticks([-1, 0, 0.65, 1, 1.325])
        cbar.set_ticklabels(["no-data", "non-class", class_name[0], class_name[1], class_name[2]])
    else:
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["no-data", "non-class", class_name])

    plt.tight_layout()
    # plt.show()

    blue_fig = plt
    return blue_fig


def fn_fp_image(target, pred):
    # if t_1=p_1: TP - if t_0=p_0: TN - if t_1=p_0: FP - if t_0>p_1: FN
    if target < pred:
        # FP = model guess P wrong
        fn_fp = 1.25
    elif target > pred:
        # FN = model guess N wrong
        fn_fp = 0.75
    else:
        # TP and TN is set kept as 1 and 0
        fn_fp = target
    return fn_fp


def runs_of_ones_list(bits):
    ones_list = [sum(g) for b, g in itertools.groupby(bits) if b]
    return ones_list


def pred_statistic(pred_image):
    pred_stat = {"artifacts":[], "mean":[], "median":[], "standard deviation":[],
                 "mean-sequence length":[], "median-sequence length":[]}

    stat_dict = {"artifacts":[], "mean":[], "median":[], "standard deviation":[],
                 "mean-sequence length":[], "median-sequence length":[]}
    for i in pred_image:
        arr = np.array(i)
        stat_dict["artifacts"].append(int(sum(i)))
        # mean
        stat_dict["mean"].append(arr.mean().round(3))
        # median
        stat_dict["median"].append(np.median(arr))
        # standard deviation
        stat_dict["standard deviation"].append(arr.std().round(3))
        # count_ratio (same as mean)

        sequnce_lens = np.array(runs_of_ones_list(i))
        if len(sequnce_lens) < 1:
            sequnce_lens = np.array([0])
        # mean-sequence length
        stat_dict["mean-sequence length"].append(sequnce_lens.mean().round(3))
        # median-sequence length
        stat_dict["median-sequence length"].append(np.median(sequnce_lens).round(3))

    for k in stat_dict:
        stat_dict[k] = np.array(stat_dict[k])

    pred_stat["artifacts"] = stat_dict["artifacts"].sum()
    pred_stat["mean"] = stat_dict["mean"].mean().round(3)
    pred_stat["median"] = np.median(stat_dict["mean"])
    pred_stat["standard deviation"] = stat_dict["standard deviation"].std().round(3)
    pred_stat["mean-sequence length"] = stat_dict["mean-sequence length"].mean().round(3)
    pred_stat["median-sequence length"] = np.median(stat_dict["median-sequence length"])

    return pred_stat


def false_pred(target, prediction):
    target = np.array(target)
    prediction = np.array(prediction)
    if np.sum(target) < 1:
        return 0
    return np.mean(prediction[target > 0])


def myloss(target, pred):
    # be mindful if ["f1_score" or "1 - f1_score"]
    return 1 - f1_score(target, pred, average='weighted')


def get_patient_id_TUAR(text):
    return text.split('_')[0]


def get_patient_ID_BC(text):
    return text.split('\\')[0].split('_')[-1]


def get_ID_BC(text):
    return text.split('\\')[0]

try:
    import xgboost
except:
    xgboost = None


## TODO: Polish -

save_path = r'C:\Users\anden\Desktop\Temp\tmp\TUAR_BC_results'

label_column = {"eyem": 0, "chew": 1, "shiv": 2, "elpp": 3, "musc": 4, "null": 5}
# load model & label classes
model_TUAR14_location = r'C:\Users\anden\PycharmProjects\results\TUAR14_null_model_045label.sav'
model_TUAR14 = pickle.load(open(model_TUAR14_location, 'rb'))
model_TUAR19_location = r'C:\Users\anden\PycharmProjects\results\TUAR19_null_model_045label.sav'
model_TUAR19 = pickle.load(open(model_TUAR19_location, 'rb'))
# model_TUAR19_location = r'C:\Users\anden\PycharmProjects\results\thesis_data_param_045label.sav'
# model_TUAR19 = pickle.load(open(model_TUAR19_location, 'rb'))
select_class = [0, 4, 5]
named_class = ['eyem', 'musc', 'null']

# load data
load_flags = [0, 0, 1]
if load_flags[0] is 1:
    # load test data _ TUAR14
    TUAR14_test = np.load(r"../../temp_mini_data/TUAR14_undersampled_test.npy", allow_pickle='TRUE')
    X_test = np.vstack(TUAR14_test[0])
    Y_test = np.vstack(TUAR14_test[1])
    name_test = np.vstack(TUAR14_test[2])
    model_names = {'TUAR14_null': model_TUAR14}
elif load_flags[1] is 1:
    # load test data _ TUAR19
    TUAR19_test = np.load(r"../../temp_mini_data/TUAR19_seed_test.npy", allow_pickle='TRUE')
    X_test = np.vstack(TUAR19_test[0])
    Y_test = np.vstack(TUAR19_test[1])
    name_test = np.vstack(TUAR19_test[2])
    # X_test = np.load(r"../../temp_mini_data/old thesis stuff/X_test_mini.npy", allow_pickle='TRUE')
    # Y_test = np.load(r"../../temp_mini_data/old thesis stuff/Y_test_mini.npy", allow_pickle='TRUE')
    model_names = {'TUAR19_null': model_TUAR19}
elif load_flags[2] is 1:
    # load test data _ BC14
    BC14_test = np.load(r"../../temp_mini_data/BC14_full_test.npy", allow_pickle='TRUE')
    BC14_X_test = np.vstack(BC14_test[0])
    BC14_Y_test = np.vstack(BC14_test[1])
    BC14_name_test = np.vstack(BC14_test[2])

    TUAR14_test = np.load(r"../../temp_mini_data/TUAR14_full_test.npy", allow_pickle='TRUE')
    X_test = np.vstack(TUAR14_test[0])
    Y_test = np.vstack(TUAR14_test[1])
    name_test = np.vstack(TUAR14_test[2])
    model_names = {'TUAR14_null': model_TUAR14}

scoring_dict = {"model names": [], "wF1": [], "acc": [], "balanced acc": [], "sens": [],
                "sens-eyem": [], "sens-musc": [], "sens-null": [],
                "acc-eyem": [], "acc-musc": [], "acc-null": []}

# model_names = {'TUAR14_null': model_TUAR14, 'TUAR19_null': model_TUAR19, 'TUAR14_full': model_TUAR14}
# model_names = {'TUAR19_null': model_TUAR19}
for i in model_names:
    score_list = score_model(model=model_names[i], Y_target=Y_test, X_pred=X_test, classes=[named_class, select_class], mod_name=i)
    for k, v in zip(scoring_dict.keys(), score_list):
        # print([k,v])
        scoring_dict[k].append(v)

# print(scoring_dict)
print('here we have the latex list:\n'+' & '.join(["$"+str(v)+"$" for v in score_list]))

# def plt_func -> Y_ax = patient_id.keys(), X_ax = nan.([freq_s * 10min]), contour_ax = patient_id.values()
# i = "TUAR14_null"
model_predict = model_names[i].predict(X_test)
all_targets = Y_test[:, select_class]
patient_dict = {}
for i in range(len(name_test)):
    patient_id = get_patient_id_TUAR(name_test[i][0])
    prediction = model_predict[i]
    target = all_targets[i]
    if patient_id not in patient_dict:
        patient_dict[patient_id] = []
    patient_dict[patient_id].append([prediction, target, false_pred(target, prediction)])

pred_index = 0
target_index = 1
false_pred_index = 2
all_patient_ids = list(patient_dict.keys())
pred_image_eyem = np.array([[v[pred_index][0] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
pred_image_musc = np.array([[v[pred_index][1] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
pred_image_null = np.array([[v[pred_index][2] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
target_image_eyem = np.array([[v[target_index][0] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
target_image_musc = np.array([[v[target_index][1] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
target_image_null = np.array([[v[target_index][2] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
false_pred_image = np.array([[v[false_pred_index] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)

diff_image_eyem = np.array([[fn_fp_image(v[target_index][0], v[pred_index][0]) for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
diff_image_musc = np.array([[fn_fp_image(v[target_index][1], v[pred_index][1]) for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
diff_image_null = np.array([[fn_fp_image(v[target_index][2], v[pred_index][2]) for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
false_pred_image = np.array([[v[false_pred_index] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)

image_array = [pred_image_eyem, target_image_eyem, diff_image_eyem,
               pred_image_musc, target_image_musc, diff_image_musc,
               pred_image_null, target_image_null, diff_image_null]

# only for experiment 3 and 4
# pred_stat_eyem = pred_statistic(pred_image_eyem)
# pred_stat_musc = pred_statistic(pred_image_musc)
# pred_stat_null = pred_statistic(pred_image_null)
print("generated images ready to blue_plot")

# colorbar discrete breaks
boundaries = [-1.5, -0.5, 0.5, 1.5]
bound_diff = [-1.5, -0.5, 0.5, 0.8, 1.15, 1.5]
t_reso = 600
save_fig = False
show_fig = True

## blue_plots for 'eyem'
cm_eyem = colors.ListedColormap(['white', 'cornflowerblue', 'lime'])
norm_eyem = colors.BoundaryNorm(boundaries, cm_eyem.N, clip=True)
# pred eyem
pred_eyem = blue_plot(pred_image_eyem, "Subject ID", "Window num [s]",
                             ["__ pred blue-plot$_{eyem}$", 'eyem'], t_reso,
                             **{"cmap":cm_eyem, "norm":norm_eyem})
if save_fig is True:
    pred_eyem.savefig(save_path+r"\_pred_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    pred_eyem.show()
# target eyem
target_eyem = blue_plot(target_image_eyem, "Subject ID", "Window num [s]",
                               ["___ target blue-plot$_{eyem}$", 'eyem'], t_reso,
                               **{"cmap":cm_eyem, "norm":norm_eyem})
if save_fig is True:
    target_eyem.savefig(save_path+r"\_target_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    target_eyem.show()
# conf eyem
cm_eyem_diff = colors.ListedColormap(['white', 'cornflowerblue', '#B7FFB7', 'lime', 'green'])
norm_eyem_diff = colors.BoundaryNorm(bound_diff, cm_eyem_diff.N, clip=True)

diff_eyem = blue_plot(diff_image_eyem, "Subject ID", "Window num [s]",
                             ["___ conf blue-plot$_{eyem}$", ['false negative', 'eyem', 'false positive']], t_reso,
                             **{"cmap":cm_eyem_diff, "norm":norm_eyem_diff})
if save_fig is True:
    diff_eyem.savefig(save_path+r"\_diff_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    diff_eyem.show()
# conf musc
cm_musc_diff = colors.ListedColormap(['white', 'cornflowerblue', 'wheat', 'yellow', 'goldenrod'])
norm_musc_diff = colors.BoundaryNorm(bound_diff, cm_musc_diff.N, clip=True)
# conf null
cm_null_diff = colors.ListedColormap(['white', 'cornflowerblue', 'darksalmon', 'red', 'maroon'])
norm_null_diff = colors.BoundaryNorm(bound_diff, cm_null_diff.N, clip=True)

print('lets wait here')



# BC test
BC14_predict = model_TUAR14.predict(BC14_X_test)
patient_dict = {}
for i in range(len(BC14_name_test)):
    # patient_id = get_patient_ID_BC(BC14_name_test[i][0])
    patient_id = get_ID_BC(BC14_name_test[i][0])
    prediction = BC14_predict[i]
    if patient_id not in patient_dict:
        patient_dict[patient_id] = []
    patient_dict[patient_id].append([prediction])

pred_index = 0
all_patient_ids = list(patient_dict.keys())
pred_image_eyem = np.array([[v[pred_index][0] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
pred_image_musc = np.array([[v[pred_index][1] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
pred_image_null = np.array([[v[pred_index][2] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)

BC14_pred_stat_eyem = pred_statistic(pred_image_eyem)
BC14_pred_stat_musc = pred_statistic(pred_image_musc)
BC14_pred_stat_null = pred_statistic(pred_image_null)

BC14_pred_eyem = blue_plot(pred_image_eyem, "Patient ID", "Window num", t_reso)
BC14_pred_eyem.title("BC14 $blue-plot_{eyem}$")
BC14_pred_eyem.savefig(save_path+r"\BC14_pred_eyem.png", format='png', bbox_inches='tight')
# BC14_pred_eyem.show()
BC14_pred_musc = blue_plot(pred_image_musc, "Patient ID", "Window num", t_reso)
BC14_pred_musc.title("BC14 $blue-plot_{musc}$")
BC14_pred_musc.savefig(save_path+r"\BC14_pred_musc.png", format='png', bbox_inches='tight')
# BC14_pred_musc.show()
BC14_pred_null = blue_plot(pred_image_null, "Patient ID", "Window num", t_reso)
BC14_pred_null.title("BC14 $blue-plot_{null}$")
BC14_pred_null.savefig(save_path+r"\BC14_pred_null.png", format='png', bbox_inches='tight')
BC14_pred_null.show(bbox_inches='tight')


# casper note for false preds
# y = [0 0 1], pred = [0 1 1], np.mean( pred[y>0])
# plot np.mean som sort/hvid

# inspect BC that might have issues
# 'sbs2data_2018_08_31_11_53_31_213' AKA /mgh/smartphone/87_DIA (2).edf OR /mgh/smartphone/87_DIA.edf


print([i.split("_")[-1]for i in all_patient_ids])