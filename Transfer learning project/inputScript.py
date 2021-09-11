import os, mne, torch, re, time
from collections import defaultdict
import numpy as np
import pandas as pd
import eegLoader
from eegProcess import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow
from scipy import signal
import matplotlib.pyplot as plt

import hpsklearn, random, glob
from hpsklearn import HyperoptEstimator, ada_boost, gaussian_nb, knn, linear_discriminant_analysis, random_forest, sgd, xgboost_classification
from hyperopt import hp
from sklearn.metrics import f1_score, recall_score
from sklearn.utils import shuffle

try:
    import xgboost
except:
    xgboost = None


## input functions
def myloss(target, pred):
    # be mindful if ["f1_score" or "1 - f1_score"]
    return 1 - f1_score(target, pred, average='weighted')

def rm_nan(x, y, name):
    X = np.vstack(x)
    Y = np.vstack(y)
    NAME = np.hstack(name)
    idx_non_nan = ~np.isnan(X).any(axis=1)
    train_x = X[idx_non_nan]
    train_y = Y[idx_non_nan]
    train_name = NAME[idx_non_nan]

    return [train_x, train_y, train_name]


def under_sample(matrix_X, matrix_Y, matrix_name, label, under_samp=30):
    Y_under_samp_idx = [idx[0] for idx in enumerate(matrix_Y) if idx[1][label]]
    Y_under_samp = list(np.split(random.sample(Y_under_samp_idx, len(Y_under_samp_idx)), [int(len(Y_under_samp_idx)/under_samp)])[1])
    mask = np.ones(matrix_Y.shape[0], dtype=bool)
    mask[Y_under_samp] = False
    result_Y = matrix_Y[mask]
    result_X = matrix_X[mask]
    result_name = matrix_name[mask]
    return [result_X, result_Y, result_name]


# how to natural sort tensor names
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


# this remove the hpsklearn warning (might ignore/remove later)
os.environ['OMP_NUM_THREADS'] = "1"
set_seed = 1615391502
random.seed(set_seed)

## handling data path and subject splits
os.chdir(os.getcwd())
save_dir = r"C:\Users\anden\PycharmProjects\NovelEEG" + "\\"  # ~~~ What is your execute path?
BC_dir = r'data_farrahtue_EEG\custom_BC_select' + "\\"
TUAR_dir = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\**\01_tcp_ar" + "\\"
data_dir = save_dir + BC_dir
BC_data = eegLoader.findEdf(path=BC_dir, selectOpt=False, saveDir=save_dir)
TUAR_data = eegLoader.findEdf(path=TUAR_dir, selectOpt=False, saveDir=save_dir)

tutorial_model = ['00008181_s003_t001.edf', '00010418_s008_t000.edf', '00009994_s001_t000.edf',
                  '00008829_s001_t002.edf', '00006096_s001_t000.edf', '00009232_s004_t009.edf',
                  '00005932_s004_t000.edf', '00010021_s001_t000.edf', '00000715_s009_t003.edf',
                  '00003573_s003_t000.edf', '00008738_s004_t001.edf', '00010023_s002_t005.edf',
                  '00006811_s001_t000.edf', '00006811_s005_t000.edf', '00007193_s001_t000.edf',
                  '00010158_s003_t000.edf', '00009232_s004_t008.edf', '00001217_s002_t000.edf',
                  '00010158_s005_t000.edf', '00010551_s002_t003.edf', '00009232_s004_t004.edf',
                  '00000768_s003_t000.edf', '00008092_s006_t001.edf', '00007383_s001_t001.edf',
                  '00007020_s001_t001.edf', '00008829_s001_t001.edf']

tutorial_subset = list(np.split(random.sample(list(TUAR_data), len(TUAR_data)), [int(.09 * len(TUAR_data))])[0])
# for all subjects run as: file_selected = TUAR_data
file_selected = TUAR_data.copy()
# pop_for_now = ['00008760_s001_t001.edf', '00008770_s001_t001.edf']
# [file_selected.pop(key) for key in pop_for_now]

# prepare TUAR output
counter = 0  # debug counter
tic_load = time.time()

# initialize paths, torch_catchers, model input X and model labels Y
TUAR14_dir = r'C:\Users\anden\Documents\PersonalProjects\TUAR14tempData' + '\\'
# TUAR19_dir = r'C:\Users\anden\Documents\PersonalProjects\TUAR19tempData' + '\\'
TUAR19_dir = r'C:\Users\anden\Documents\PersonalProjects\TUAR19_new_24hztempData' + '\\'
BC14_dir = r'C:\Users\anden\Documents\PersonalProjects\BC14tempData' + '\\'
tensor_dir = TUAR14_dir
label_column = {"eyem": 0, "chew": 1, "shiv": 2, "elpp": 3, "musc": 4, "null": 5}
counter_subject = {"eyem": set(), "chew": set(), "shiv": set(), "elpp": set(), "musc": set(), "null": set()}
counter_session = {"eyem": set(), "chew": set(), "shiv": set(), "elpp": set(), "musc": set(), "null": set()}
counter_recording = {"eyem": set(), "chew": set(), "shiv": set(), "elpp": set(), "musc": set(), "null": set()}
counter_second = {"eyem": 0, "chew": 0, "shiv": 0, "elpp": 0, "musc": 0, "null": 0}

data_splits = np.split(random.sample(list(file_selected), len(file_selected)), [int(.6 * len(file_selected)),
                                                                                int(.8 * len(file_selected))])
data_splits_test = data_splits #[[], [], np.concatenate(data_splits, axis=0)]
stack_X_train = list()
stack_Y_train = list()
stack_name_train = list()
stack_X_val = list()
stack_Y_val = list()
stack_name_val = list()
stack_X_test = list()
stack_Y_test = list()
stack_name_test = list()

subjects = defaultdict(dict)
if __name__ == '__main__':
    for edf in file_selected:  # TUAR_data:
        subject_ID = edf.split('_')[0]
        if subject_ID in subjects.keys():
            subjects[subject_ID][edf] = TUAR_data[edf].copy()
        else:
            subjects[subject_ID] = {edf: TUAR_data[edf].copy()}

        # debug counter for subject error
        counter += 1
        print("\n%s is session: %i" % (edf, counter))
        # if counter == 135: #TODO: remove
        #     continue

        # initialize hierarchical dict
        proc_subject = subjects[subject_ID][edf]

        proc_subject["label_loader"] = []
        proc_subject["input_loader"] = []
        proc_subject["window_name"] = []

        tensor_paths = glob.glob(tensor_dir + edf.split(".")[0] + "\\" + "**/*.pt", recursive=True)
        tensor_paths.sort(key=natural_keys)

        for pt_dir in tensor_paths:
            try:
                tmp_loader = torch.load(pt_dir)
                tmp_label = np.zeros(len(label_column))
                tmp_name = '\\'.join(pt_dir.split('\\')[-2:])
                if len(tmp_loader[1]) != 1 and 'null' in tmp_loader[1]:
                    tmp_loader[1].remove('null')
                for label in tmp_loader[1]:
                    tmp_label[label_column[label]] = 1
                    counter_subject[label].add(pt_dir.split("\\")[-2].split("_")[0])
                    counter_session[label].add(pt_dir.split("\\")[-2].split('_t')[0])
                    counter_recording[label].add(pt_dir.split("\\")[-2])
                    counter_second[label] += 1
                proc_subject["label_loader"].append(tmp_label.astype(np.int8))
                proc_subject["input_loader"].append(tmp_loader[0].numpy().flatten().astype(np.float16))
                proc_subject["window_name"].append(tmp_name)
            except:
                print('break debug')
                proc_subject["label_loader"].append(tmp_label.astype(np.int8))
                proc_subject["input_loader"].append(tmp_loader[0].numpy().flatten().astype(np.float16))
                proc_subject["window_name"].append(tmp_name)
                continue

        # This might be lower RAM usage by executing as list-comprehensions outside loop
        if edf in data_splits_test[0]:
            stack_X_train.append(np.stack(proc_subject["input_loader"]))
            stack_Y_train.append(np.stack(proc_subject["label_loader"]))
            stack_name_train.append(np.stack(proc_subject["window_name"]))
        elif edf in data_splits_test[1]:
            stack_X_val.append(np.stack(proc_subject["input_loader"]))
            stack_Y_val.append(np.stack(proc_subject["label_loader"]))
            stack_name_val.append(np.stack(proc_subject["window_name"]))
        elif edf in data_splits_test[2]:
            stack_X_test.append(np.stack(proc_subject["input_loader"]))
            stack_Y_test.append(np.stack(proc_subject["label_loader"]))
            stack_name_test.append(np.stack(proc_subject["window_name"]))
        else:
            print("break pause")

    # train_pure = np.array([stack_X_train, stack_Y_train, stack_name_train], dtype=object)
    # val_pure = np.array([stack_X_val, stack_Y_val, stack_name_val], dtype=object)
    # test_pure = np.array([stack_X_test, stack_Y_test, stack_name_test], dtype=object)

    train_x, train_y, train_name = rm_nan(stack_X_train, stack_Y_train, stack_name_train) #TODO: remove #
    # train_x, train_y, train_name = under_sample(train_x, train_y, train_name, label=5, under_samp=30) #TODO: remove #

    val_x, val_y, val_name = rm_nan(stack_X_val, stack_Y_val, stack_name_val) #TODO: remove #
    # val_x, val_y, val_name = under_sample(val_x, val_y, val_name, label=5, under_samp=30) #TODO: remove #

    # ch_error_BC = [i for i, l in enumerate(stack_X_test) if l.shape[1] == 350]
    # x_foo = list()
    # y_foo = list()
    # name_foo = list()
    # for i in ch_error_BC:
    #     x_foo.append(stack_X_test[i])
    #     y_foo.append(stack_Y_test[i])
    #     name_foo.append(stack_name_test[i])
    #
    # test_x, test_y, test_name = rm_nan(x_foo, y_foo, name_foo)
    #

    test_x, test_y, test_name = rm_nan(stack_X_test, stack_Y_test, stack_name_test)
    # test_x, test_y, test_name = under_sample(test_x, test_y, test_name, label=5, under_samp=30) #TODO: remove #

    save_np_flag = True
    if save_np_flag is True:
        numpy_saves = r'C:\Users\anden\PycharmProjects\temp_mini_data\TUAR14_full'
        # new method (please get to this)
        # train = np.array([stack_X_train, stack_Y_train, stack_name_train], dtype=object)
        # val = np.array([stack_X_val, stack_Y_val, stack_name_val], dtype=object)
        # test = np.array([stack_X_test, stack_Y_test, stack_name_test], dtype=object)
        # old method (rewrite)
        train = np.array([list(train_x), list(train_y), list(train_name)], dtype=object) #TODO: remove #
        val = np.array([list(val_x), list(val_y), list(val_name)], dtype=object) #TODO: remove #
        test = np.array([list(test_x), list(test_y), list(test_name)], dtype=object)
        np.save(numpy_saves + r'_train.npy', train) #TODO: remove #
        np.save(numpy_saves + r'_val.npy', val) #TODO: remove #
        np.save(numpy_saves + r'_test.npy', test)

    # BC14_test = np.load(numpy_saves + r'_test_full.npy', allow_pickle='TRUE')

    counter_table = dict()
    for key in label_column:
        counter_table[key] = [len(counter_subject[key]), len(counter_session[key]),
                              len(counter_recording[key]), counter_second[key]]
    # list form -> list(counter_table.items())

    toc_load = time.time()
    print("\n~~~~~~~~~~~~~~~~~~~~\n"
          "it took %imin:%is to load tensors for %i patients"
          "\n~~~~~~~~~~~~~~~~~~~~\n"
          % (int((toc_load - tic_load) / 60), int((toc_load - tic_load) % 60), len(subjects)))

    print(counter_table)

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "it took %imin:%is to load:\n"
          " X_train = %s with Y_train = %s\n"
          " X_val = %s with Y_val = %s\n"
          " X_test = %s with Y_test = %s\n\n" #          " X_model = %s with Y_model = %s\n"
          "%s"
          "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
          % (int((toc_load - tic_load) / 60), int((toc_load - tic_load) % 60),
             train_x.shape, train_y.shape,
             val_x.shape, val_y.shape,
             test_x.shape, test_y.shape,             # X_model.shape, Y_model.shape,
             np.array([train_y.sum(axis=0), val_y.sum(axis=0), test_y.sum(axis=0)])))

    print("end of script - before model testing")

# TODO:
