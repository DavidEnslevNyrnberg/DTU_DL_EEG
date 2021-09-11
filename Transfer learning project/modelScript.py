# Script
# This script/function/class should:
# load a set of pre-split data and stack a train/val data to a model data set.
# The given dataset should be used to build LDA model of 6 one-vs-rest classifier
# >pickle< should be used to safe/load the model across scripts for scoring
# DAVID NOTE: do not need to load test data her (just train and val)

# system imports
import os, time, random, pickle, glob, re, platform
# EEG imports
import mne, torch, hpsklearn
import eegLoader
import numpy as np
import pandas as pd

from hpsklearn import ada_boost, gaussian_nb, knn, random_forest, sgd, xgboost_classification
from hpsklearn import HyperoptEstimator, linear_discriminant_analysis, one_vs_rest
from hyperopt import hp
from sklearn.metrics import f1_score, recall_score, classification_report, balanced_accuracy_score, accuracy_score

try:
    import xgboost
except:
    xgboost = None

# remove hpsklearn warning (might ignore/remove later)
os.environ['OMP_NUM_THREADS'] = "1"

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
save_score_path = r'C:\Users\anden\PycharmProjects\results'
save_dir = r"C:/Users/anden/PycharmProjects/NovelEEG" + "//"  # ~~~ What is your execute path?
numpy_dir = r"C:/Users/anden/PycharmProjects/temp_mini_data" + "//"
# TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset" + "//"  # /**/01_tcp_ar #/100/00010023/s002_2013_02_21
TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset/**/01_tcp_ar" + "//"  # debug 01_tcp_ar
# TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset/**/02_tcp_le"+"//" # debug 02_tcp_le
# TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset/**/03_tcp_ar_a"+"//" # debug 03_tcp_ar_a
TUAR_dirDir = save_dir + TUAR_dir

# seed finder int(time.time())
seed_val = 1615391502  # 10 is OK 1615391502 is good
# seed_val = time.time()
random.seed(seed_val)
label_column = {"eyem": 0, "chew": 1, "shiv": 2, "elpp": 3, "musc": 4, "null": 5}

def myloss(target, pred):
    # be mindful if ["f1_score" or "1 - f1_score"]
    return 1 - f1_score(target, pred, average='weighted')


def bench_classifiers(name):
    classifiers = [
        ada_boost(name + '.ada_boost'),  # boo
        gaussian_nb(name + '.gaussian_nb'),  # eey
        knn(name + '.knn', sparse_data=True),  # eey
        linear_discriminant_analysis(name + '.linear_discriminant_analysis', n_components=1),  # eey
        random_forest(name + '.random_forest'),  # boo
        sgd(name + '.sgd')  # eey
    ]
    if xgboost:
        classifiers.append(xgboost_classification(name + '.xgboost'))  # boo
    return hp.choice('%s' % name, classifiers)


if __name__ == '__main__':
    try:
        tic_load = time.time()
        X_train_mini = np.load(r"../../temp_mini_data/old thesis stuff/X_train_mini.npy", allow_pickle='TRUE')
        Y_train_mini = np.load(r"../../temp_mini_data/old thesis stuff/Y_train_mini.npy", allow_pickle='TRUE')
        # name_train_mini = np.load(r"../../temp_mini_data/old thesis stuff/tuar14_train_name.npy", allow_pickle='TRUE')
        X_val_mini = np.load(r"../../temp_mini_data/old thesis stuff/X_val_mini.npy", allow_pickle='TRUE')
        Y_val_mini = np.load(r"../../temp_mini_data/old thesis stuff/Y_val_mini.npy", allow_pickle='TRUE')
        # name_val_mini = np.load(r"../../temp_mini_data/old thesis stuff/tuar14_val_name.npy", allow_pickle='TRUE')
        X_test_mini = np.load(r"../../temp_mini_data/old thesis stuff/X_test_mini.npy", allow_pickle='TRUE')
        Y_test_mini = np.load(r"../../temp_mini_data/old thesis stuff/Y_test_mini.npy", allow_pickle='TRUE')
        # name_test_mini = np.load(r"../../temp_mini_data/old thesis stuff/tuar14_test_name.npy", allow_pickle='TRUE')
        # X_train_mini = np.load(r"temp_mini_data/X_train_mini.npy", allow_pickle='TRUE')
        # Y_train_mini = np.load(r"temp_mini_data/Y_train_mini.npy", allow_pickle='TRUE')
        # X_val_mini = np.load(r"temp_mini_data/X_val_mini.npy", allow_pickle='TRUE')
        # Y_val_mini = np.load(r"temp_mini_data/Y_val_mini.npy", allow_pickle='TRUE')
        # X_test_mini = np.load(r"temp_mini_data/X_test_mini.npy", allow_pickle='TRUE')
        # Y_test_mini = np.load(r"temp_mini_data/Y_test_mini.npy", allow_pickle='TRUE')

        names = ["thesis_data"]
        for name in names:
            # train = np.load(r"../../temp_mini_data/"+name+"_train.npy", allow_pickle='TRUE')
            # val = np.load(r"../../temp_mini_data/"+name+"_val.npy", allow_pickle='TRUE')
            # test = np.load(r"../../temp_mini_data/"+name+"_test.npy", allow_pickle='TRUE')
            orig_data_dist = np.load(r"../../temp_mini_data/orig_data_dist.npy", allow_pickle='TRUE')

            # MEME TODO: .tolist() sets EVERYTHING to list
            # X_train_mini, Y_train_mini, name_train_mini = [np.vstack(train[0]), np.vstack(train[1]), np.vstack(train[2])]
            # X_val_mini, Y_val_mini, name_val_mini = [np.vstack(val[0]), np.vstack(val[1]), np.vstack(val[2])]
            # X_test_mini, Y_test_mini, name_test_mini = [np.vstack(test[0]), np.vstack(test[1]), np.vstack(test[2])]

            X_model_mini = np.concatenate((X_train_mini, X_val_mini))
            Y_model_mini = np.concatenate((Y_train_mini, Y_val_mini))

            toc_load = time.time()
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                  "Original data:\n%s"
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" % orig_data_dist)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                  "it took %imin:%is to load:\n"
                  " X_train = %s with Y_train = %s\n"
                  " X_val = %s with Y_val = %s\n\n"
                  " X_model = %s with Y_model = %s\n"
                  "%s"
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                  % (int((toc_load - tic_load) / 60), int((toc_load - tic_load) % 60),
                     X_train_mini.shape, Y_train_mini.shape,
                     X_val_mini.shape, Y_val_mini.shape,
                     X_model_mini.shape, Y_model_mini.shape,
                     np.array([Y_train_mini.sum(axis=0), Y_val_mini.sum(axis=0), Y_test_mini.sum(axis=0)])))

            print("\ndata is loaded  - next step > model testing\n")

            n_job = 5
            select_classes = [0, 4, 5]  # eyem, musc, null
            val_dist = X_val_mini.shape[0] / X_train_mini.shape[0]

            tic_mod_all = time.time()
            # select_alg = [ada_boost(name + '.ada_boost'),
            #               gaussian_nb(name + '.gaussian_nb'),
            #               knn(name + '.knn', sparse_data=True),
            #               linear_discriminant_analysis(name + '.linear_discriminant_analysis', n_components=1),
            #               random_forest(name + '.random_forest'),
            #               sgd(name + '.sgd'),
            #               xgboost_classification(name + '.xgboost')]
            select_alg = [linear_discriminant_analysis(name + '_LDA', n_components=1)]

            # fitting models
            estim_one_vs_rest = dict()
            # scoring models
            algo_scoring = dict()
            for alg in select_alg:
                tic_mod = time.time()
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                      "running on %s" % (alg.name + '_one_V_all'),
                      "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                clf_method = one_vs_rest(str(alg.name + '_one_V_all'), estimator=alg, n_jobs=n_job)
                estim_one_vs_rest[alg.name + '_one_V_all'] = HyperoptEstimator(classifier=clf_method, loss_fn=myloss,
                                                                               n_jobs=n_job)
                estim_one_vs_rest[alg.name + '_one_V_all'].fit(X_model_mini, Y_model_mini[:, select_classes],
                                                               valid_size=val_dist, cv_shuffle=False, random_state=seed_val)
                # filename = r'C:\Users\anden\PycharmProjects\results\TUAR_14ch_param_cl045.sav'
                model = estim_one_vs_rest[alg.name + '_one_V_all']
                pickle.dump(model, open('\\'.join([save_score_path, name]) + r'_param_045label.sav', 'wb'))

                toc_mod = time.time()
                print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                      "fitting model took %ih:%imin:%is"
                      "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                      % (int((toc_mod - tic_mod) / 60 / 60),
                         int((toc_mod - tic_mod) / 60) - 60 * int((toc_mod - tic_mod) / 60 / 60),
                         int((toc_mod - tic_mod) % 60)))

    except:
        print('error occured')

