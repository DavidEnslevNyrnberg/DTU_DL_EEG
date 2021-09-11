# system imports
import os, time, random, pickle, glob, re, platform, itertools
# EEG imports
import mne, torch, hpsklearn
import eegLoader
import numpy as np
import pandas as pd
import seaborn as sns
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
    acc = 1 - model.score(X_pred, Y_target[:, classes[1]])
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
    pred_stat = {"artifacts":[], "median":[], "mean":[], "standard deviation":[],
                 "mean-sequence length":[], "median-sequence length":[]}

    stat_dict = {"artifacts":[], "median":[], "mean":[], "standard deviation":[],
                 "mean-sequence length":[], "median-sequence length":[]}
    for i in pred_image:
        arr = np.array(i)
        stat_dict["artifacts"].append(int(sum(i)))
        # median
        stat_dict["median"].append(np.median(arr))
        # mean
        stat_dict["mean"].append(arr.mean().round(3))
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
    pred_stat["median"] = np.median(stat_dict["mean"]).round(3)
    pred_stat["mean"] = stat_dict["mean"].mean().round(3)
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


save_path = r'C:\Users\anden\Desktop\Temp\tmp\TUAR_BC_results'

label_column = {"eyem": 0, "chew": 1, "shiv": 2, "elpp": 3, "musc": 4, "null": 5}
# load model & label classes
model_TUAR14_location = r'C:\Users\anden\PycharmProjects\results\TUAR14_null_model_045label.sav'
model_TUAR14 = pickle.load(open(model_TUAR14_location, 'rb'))
select_class = [0, 4, 5]
named_class = ['eyem', 'musc', 'null']

## Experiment 3 notes: Wants from TUAR14_full: table, blue_plot(pred), blue_plot(target), blue_plot(fn_fp), descriptive_stat()
#       Wants from BC14: blue_plot(pred), blue_plot(target), blue_plot(fn_fp), descriptive_stat()
BC14_test = np.load(r"../../temp_mini_data/BC14_full_test.npy", allow_pickle='TRUE')
BC14_X_test = np.vstack(BC14_test[0])
BC14_Y_test = np.vstack(BC14_test[1])
BC14_name_test = np.vstack(BC14_test[2])
TUAR14_test = np.load(r"../../temp_mini_data/TUAR14_full_test.npy", allow_pickle='TRUE')
TUAR14_test_X = np.vstack(TUAR14_test[0])
TUAR14_test_Y = np.vstack(TUAR14_test[1])
TUAR14_test_name = np.vstack(TUAR14_test[2])
model_names = {'TUAR14_null': model_TUAR14}

## tabel scoring:
scoring_dict = {"model names": [], "wF1": [], "acc": [], "balanced acc": [], "sens": [],
                "sens-eyem": [], "sens-musc": [], "sens-null": [],
                "acc-eyem": [], "acc-musc": [], "acc-null": []}

for i in model_names:
    score_list = score_model(model=model_names[i], Y_target=TUAR14_test_Y, X_pred=TUAR14_test_X, classes=[named_class, select_class], mod_name=i)
    for k, v in zip(scoring_dict.keys(), score_list):
        # print([k,v])
        scoring_dict[k].append(v)

# print(scoring_dict)
print('here we have the latex list:\n'+' & '.join(["$"+str(v)+"$" for v in score_list]))

## TUAR14 blue_plot scoring:
model_predict = model_TUAR14.predict(TUAR14_test_X)
all_targets = TUAR14_test_Y[:, select_class]
patient_dict = {}
for i in range(len(TUAR14_test_name)):
    patient_id = get_patient_id_TUAR(TUAR14_test_name[i][0])
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

diff_image_eyem = np.array([[fn_fp_image(v[target_index][0], v[pred_index][0]) for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
diff_image_musc = np.array([[fn_fp_image(v[target_index][1], v[pred_index][1]) for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
diff_image_null = np.array([[fn_fp_image(v[target_index][2], v[pred_index][2]) for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
false_pred_image = np.array([[v[false_pred_index] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)

TUAR14_image_array = [pred_image_eyem, target_image_eyem, diff_image_eyem,
               pred_image_musc, target_image_musc, diff_image_musc,
               pred_image_null, target_image_null, diff_image_null]

# only for experiment 3 and 4
TUAR14_pred_stat_eyem = pred_statistic(pred_image_eyem)
TUAR14_pred_stat_musc = pred_statistic(pred_image_musc)
TUAR14_pred_stat_null = pred_statistic(pred_image_null)
print(re.sub(',', ' &', str(list(TUAR14_pred_stat_eyem.values()))))
print(re.sub(',', ' &', str(list(TUAR14_pred_stat_musc.values()))))
print(re.sub(',', ' &', str(list(TUAR14_pred_stat_null.values()))))

print("generated images ready to blue_plot")
# colorbar discrete breaks
boundaries = [-1.5, -0.5, 0.5, 1.5]
bound_diff = [-1.5, -0.5, 0.5, 0.8, 1.15, 1.5]
t_reso = 600
save_fig = True
show_fig = False

## blue_plots for 'eyem'
cm_eyem = colors.ListedColormap(['white', 'cornflowerblue', 'lime'])
norm_eyem = colors.BoundaryNorm(boundaries, cm_eyem.N, clip=True)
# pred eyem
TUAR14_pred_eyem = blue_plot(pred_image_eyem, "Subject ID", "Window num [s]",
                             ["TUAR14 pred blue-plot$_{eyem}$", 'eyem'], t_reso,
                             **{"cmap":cm_eyem, "norm":norm_eyem})
if save_fig is True:
    TUAR14_pred_eyem.savefig(save_path+r"\ex3_TUAR14_pred_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_pred_eyem.show()
# target eyem
TUAR14_target_eyem = blue_plot(target_image_eyem, "Subject ID", "Window num [s]",
                               ["TUAR14 target blue-plot$_{eyem}$", 'eyem'], t_reso,
                               **{"cmap":cm_eyem, "norm":norm_eyem})
if save_fig is True:
    TUAR14_target_eyem.savefig(save_path+r"\ex3_TUAR14_target_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_target_eyem.show()
# confusion eyem
cm_eyem_diff = colors.ListedColormap(['white', 'cornflowerblue', '#B7FFB7', 'lime', 'green'])
norm_eyem_diff = colors.BoundaryNorm(bound_diff, cm_eyem_diff.N, clip=True)

TUAR14_diff_eyem = blue_plot(diff_image_eyem, "Subject ID", "Window num [s]",
                             ["TUAR14 confusion blue-plot$_{eyem}$", ['false negative', 'eyem', 'false positive']], t_reso,
                             **{"cmap":cm_eyem_diff, "norm":norm_eyem_diff})
if save_fig is True:
    TUAR14_diff_eyem.savefig(save_path+r"\ex3_TUAR14_diff_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_diff_eyem.show()

## blue_plots for 'musc'
cm_musc = colors.ListedColormap(['white', 'cornflowerblue', 'yellow'])
norm_musc = colors.BoundaryNorm(boundaries, cm_musc.N, clip=True)
# pred musc
TUAR14_pred_musc = blue_plot(pred_image_musc, "Subject ID", "Window num [s]",
                             ["TUAR14 pred blue-plot$_{musc}$", 'musc'], t_reso,
                             **{"cmap":cm_musc, "norm":norm_musc})
if save_fig is True:
    TUAR14_pred_musc.savefig(save_path+r"\ex3_TUAR14_pred_musc.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_pred_musc.show()
# target musc
TUAR14_target_musc = blue_plot(target_image_musc, "Subject ID", "Window num [s]",
                               ["TUAR14 target blue-plot$_{musc}$", 'musc'], t_reso,
                               **{"cmap":cm_musc, "norm":norm_musc})
if save_fig is True:
    TUAR14_target_musc.savefig(save_path+r"\ex3_TUAR14_target_musc.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_target_musc.show()
# confusion musc
cm_musc_diff = colors.ListedColormap(['white', 'cornflowerblue', 'wheat', 'yellow', 'goldenrod'])
norm_musc_diff = colors.BoundaryNorm(bound_diff, cm_musc_diff.N, clip=True)

TUAR14_diff_musc = blue_plot(diff_image_musc, "Subject ID", "Window num [s]",
                             ["TUAR14 confusion blue-plot$_{null}$", ['false negative', 'musc', 'false positive']], t_reso,
                             **{"cmap":cm_musc_diff, "norm":norm_musc_diff})
if save_fig is True:
    TUAR14_diff_musc.savefig(save_path+r"\ex3_TUAR14_diff_musc.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_diff_musc.show()

## blue_plots for 'null'
cm_null = colors.ListedColormap(['white', 'cornflowerblue', 'red'])
norm_null = colors.BoundaryNorm(boundaries, cm_null.N, clip=True)
# pred null
TUAR14_pred_null = blue_plot(pred_image_null, "Subject ID", "Window num [s]",
                             ["TUAR14 pred blue-plot$_{null}$", 'null'], t_reso,
                             **{"cmap":cm_null, "norm":norm_null})
if save_fig is True:
    TUAR14_pred_null.savefig(save_path+r"\ex3_TUAR14_pred_null.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_pred_null.show()
# target null
TUAR14_target_null = blue_plot(target_image_null, "Subject ID", "Window num [s]",
                               ["TUAR14 target blue-plot$_{null}$", 'null'], t_reso,
                               **{"cmap":cm_null, "norm":norm_null})
if save_fig is True:
    TUAR14_target_null.savefig(save_path+r"\ex3_TUAR14_target_null.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_target_null.show()
# confusion null
cm_null_diff = colors.ListedColormap(['white', 'cornflowerblue', 'darksalmon', 'red', 'maroon'])
norm_null_diff = colors.BoundaryNorm(bound_diff, cm_null_diff.N, clip=True)

TUAR14_diff_eyem = blue_plot(diff_image_null, "Subject ID", "Window num [s]",
                             ["TUAR14 confusion blue-plot$_{null}$", ['false negative', 'null', 'false positive']], t_reso,
                             **{"cmap":cm_null_diff, "norm":norm_null_diff})
if save_fig is True:
    TUAR14_diff_eyem.savefig(save_path+r"\ex3_TUAR14_diff_null.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    TUAR14_diff_eyem.show()

print("That was the TUAR14_full - on to BC14")

BC14_predict = model_TUAR14.predict(BC14_X_test)
patient_dict = {}
for i in range(len(BC14_name_test)):
    patient_id = get_ID_BC(BC14_name_test[i][0])
    prediction = BC14_predict[i]
    if patient_id not in patient_dict:
        patient_dict[patient_id] = []
    patient_dict[patient_id].append([prediction])

quality_dir = r"C:\Users\anden\PycharmProjects\NovelEEG\Exampels\BCqualityLabel_newIDs.txt"
quality_labels = pd.read_csv(quality_dir, delimiter=",").values
quality = {qual[0]: int(qual[1]) for qual in quality_labels if qual[1].isdigit()}
pred_index = 0
all_patient_ids = [ids for ids in list(patient_dict.keys()) if ids in quality.keys()]
# all_patient_ids2 = list(patient_dict.keys())
pred_image_eyem = np.array([[v[pred_index][0] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
pred_image_musc = np.array([[v[pred_index][1] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)
pred_image_null = np.array([[v[pred_index][2] for v in patient_dict[pid]] for pid in all_patient_ids], dtype=object)

trend_dataframe = pd.DataFrame.from_dict(quality, orient='index', columns=['Quality'])
trend_dataframe['no. Eyem'] = [sum(eyem_i) for eyem_i in pred_image_eyem]
trend_dataframe['no. Musc'] = [sum(musc_i) for musc_i in pred_image_musc]
trend_dataframe['no. Null'] = [sum(musc_i) for musc_i in pred_image_null]
trend_dataframe['no. Artifacts'] = [sum(eyem_arr) + sum(pred_image_musc[i]) for i, eyem_arr in enumerate(pred_image_eyem)]

# artifact trend
trend_art_plot = sns.jointplot(x='no. Artifacts', y='Quality', data=trend_dataframe, kind='reg')
print(np.corrcoef(np.fromiter(quality.values(), dtype=int), np.array(trend_dataframe['no. Artifacts'])))
if save_fig is True:
    trend_art_plot.savefig(save_path+r"\ex4_trend_arti.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    plt.show()
# eyem trend
trend_eyem_plot = sns.jointplot(x='no. Eyem', y='Quality', data=trend_dataframe, kind='reg')
print(np.corrcoef(np.fromiter(quality.values(), dtype=int), np.array(trend_dataframe['no. Eyem'])))
if save_fig is True:
    trend_eyem_plot.savefig(save_path+r"\ex4_trend_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    plt.show()
# musc trend
trend_musc_plot = sns.jointplot(x='no. Musc', y='Quality', data=trend_dataframe, kind='reg')
print(np.corrcoef(np.fromiter(quality.values(), dtype=int), np.array(trend_dataframe['no. Musc'])))
if save_fig is True:
    trend_musc_plot.savefig(save_path+r"\ex4_trend_musc.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    plt.show()
# null trend
trend_null_plot = sns.jointplot(x='no. Null', y='Quality', data=trend_dataframe, kind='reg')
print(np.corrcoef(np.fromiter(quality.values(), dtype=int), np.array(trend_dataframe['no. Null'])))
if save_fig is True:
    trend_null_plot.savefig(save_path+r"\ex4_trend_null.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    plt.show()

BC14_pred_stat_eyem = pred_statistic(pred_image_eyem)
BC14_pred_stat_musc = pred_statistic(pred_image_musc)
BC14_pred_stat_null = pred_statistic(pred_image_null)

print(re.sub(',', ' &', str(list(BC14_pred_stat_eyem.values()))))
print(re.sub(',', ' &', str(list(BC14_pred_stat_musc.values()))))
print(re.sub(',', ' &', str(list(BC14_pred_stat_null.values()))))

## BC14 blue plots
# pred eyem
BC14_pred_eyem = blue_plot(pred_image_eyem, "Subject ID", "Window num [s]",
                           ["BC14 pred blue-plot$_{eyem}$", 'eyem'], t_reso,
                           **{"cmap":cm_eyem, "norm":norm_eyem})
if save_fig is True:
    BC14_pred_eyem.savefig(save_path+r"\ex3_BC14_pred_eyem.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    BC14_pred_eyem.show()
# pred musc
BC14_pred_musc = blue_plot(pred_image_musc, "Subject ID", "Window num [s]",
                           ["BC14 pred blue-plot$_{musc}$", 'musc'], t_reso,
                           **{"cmap": cm_musc, "norm": norm_musc})
if save_fig is True:
    BC14_pred_musc.savefig(save_path + r"\ex3_BC14_pred_musc.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    BC14_pred_musc.show()
# pred null
BC14_pred_null = blue_plot(pred_image_null, "Subject ID", "Window num [s]",
                           ["BC14 pred blue-plot$_{null}$", 'null'], t_reso,
                           **{"cmap":cm_null, "norm":norm_null})
if save_fig is True:
    BC14_pred_null.savefig(save_path+r"\ex3_BC14_pred_null.pdf", format='pdf', bbox_inches='tight')
if show_fig is True:
    BC14_pred_null.show()

print("code is done")
