import os, mne, torch, re, time, platform
from collections import defaultdict
import numpy as np
import pandas as pd
# from ../LoadFarrahTueData.loadData import jsonLoad
import loadData
from preprocessPipeline import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow
from scipy import signal
import matplotlib.pyplot as plt

import argparse
import hpsklearn, random, glob
from hpsklearn import HyperoptEstimator, ada_boost, gaussian_nb, knn, linear_discriminant_analysis, random_forest, sgd, \
    xgboost_classification, one_vs_rest
from hyperopt import hp
from sklearn.metrics import f1_score, recall_score, classification_report, balanced_accuracy_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

try:
    import xgboost
except:
    xgboost = None

# remove hpsklearn warning (might ignore/remove later)
os.environ['OMP_NUM_THREADS'] = "1"

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
save_dir = r"C:/Users/anden/PycharmProjects/NovelEEG" + "//"  # ~~~ What is your execute path?
numpy_dir = r"C:/Users/anden/PycharmProjects/temp_mini_data" + "//"
# TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset" + "//"  # /**/01_tcp_ar #/100/00010023/s002_2013_02_21
TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset/**/01_tcp_ar" + "//"  # debug 01_tcp_ar
# TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset/**/02_tcp_le"+"//" # debug 02_tcp_le
# TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset/**/03_tcp_ar_a"+"//" # debug 03_tcp_ar_a
TUAR_dirDir = save_dir + TUAR_dir

# seed finder int(time.time())
seed_val = time.time()  # 10 is OK 1615391502 is good
random.seed(seed_val)
# maybe_seed: [1615390300, 1615391502, 10]
label_column = {"eyem": 0, "chew": 1, "shiv": 2, "elpp": 3, "musc": 4, "null": 5}


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


def myloss(target, pred):
    # be mindful if ["f1_score" or "1 - f1_score"]
    return 1 - f1_score(target, pred, average='weighted')


def under_sample(matrix_X, matrix_Y, label, under_samp=30):
    Y_under_samp_idx = [idx[0] for idx in enumerate(matrix_Y) if idx[1][label]]
    Y_under_samp = list(np.split(random.sample(Y_under_samp_idx, len(Y_under_samp_idx)), [int(len(Y_under_samp_idx)/under_samp)])[1])
    mask = np.ones(matrix_Y.shape[0], dtype=bool)
    mask[Y_under_samp] = False
    result_Y = matrix_Y[mask]
    result_X = matrix_X[mask]
    return [result_X, result_Y]


if __name__ == '__main__':
    machine_info = {"machine": platform.machine(), "version": platform.version(), "platform": platform.platform(),
                    "uname": platform.uname(), "system": platform.system(), "precessor": platform.processor()}

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--subsample', default=1, type=float)
    parser.add_argument('--var_rerun', default=0, type=int)

    args = parser.parse_args()

    print(args.index, args.subsample)
    try:
        tic_load = time.time()
        X_train_mini = np.load(r"temp_mini_data/X_train_mini.npy", allow_pickle='TRUE')
        Y_train_mini = np.load(r"temp_mini_data/Y_train_mini.npy", allow_pickle='TRUE')
        X_val_mini = np.load(r"temp_mini_data/X_val_mini.npy", allow_pickle='TRUE')
        Y_val_mini = np.load(r"temp_mini_data/Y_val_mini.npy", allow_pickle='TRUE')
        X_test_mini = np.load(r"temp_mini_data/X_test_mini.npy", allow_pickle='TRUE')
        Y_test_mini = np.load(r"temp_mini_data/Y_test_mini.npy", allow_pickle='TRUE')
        orig_data_dist = np.load(r"temp_mini_data/orig_data_dist.npy", allow_pickle='TRUE')

        X_train_mini, Y_train_mini = under_sample(X_train_mini, Y_train_mini, label=0, under_samp=1/args.subsample)
        
        X_model_mini = np.concatenate((X_train_mini, X_val_mini))
        Y_model_mini = np.concatenate((Y_train_mini, Y_val_mini))

        toc_load = time.time()
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
              "Original data:\n%s"
              "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" % orig_data_dist)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
              "it took %imin:%is to load:\n"
              " X_train = %s with Y_train = %s\n"
              " X_val = %s with Y_val = %s\n"
              " X_test = %s with Y_test = %s\n\n"
              " X_model = %s with Y_model = %s\n"
              "%s"
              "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              % (int((toc_load - tic_load) / 60), int((toc_load - tic_load) % 60),
                 X_train_mini.shape, Y_train_mini.shape,
                 X_val_mini.shape, Y_val_mini.shape,
                 X_test_mini.shape, Y_test_mini.shape,
                 X_model_mini.shape, Y_model_mini.shape,
                 np.array([Y_train_mini.sum(axis=0), Y_val_mini.sum(axis=0), Y_test_mini.sum(axis=0)])))
        print(seed_val)

        print("\ndata is loaded  - next step > model testing\n")

        n_job = 6
        select_classes = [0, 1, 2, 3, 4, 5]
        val_dist = X_val_mini.shape[0] / X_train_mini.shape[0]
        name = 'my_est_oVa'

        tic_mod_all = time.time()
        select_alg = [ada_boost(name + '.ada_boost'),
                      gaussian_nb(name + '.gaussian_nb'),
                      knn(name + '.knn', sparse_data=True),
                      linear_discriminant_analysis(name + '.linear_discriminant_analysis', n_components=1),
                      random_forest(name + '.random_forest'),
                      sgd(name + '.sgd'),
                      xgboost_classification(name + '.xgboost')]
        
        # fitting models
        estim_one_vs_rest = dict()
        # scoring models
        algo_scoring = dict()
        save_score_path = r'C:/Users/anden/PycharmProjects/NovelEEG/results'
        for alg in [select_alg[args.index]]:
            tic_mod = time.time()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                  "running on %s" % (alg.name + '.one_V_all'),
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            clf_method = one_vs_rest(str(alg.name + '.one_V_all'), estimator=alg, n_jobs=1)
            estim_one_vs_rest[alg.name + '.one_V_all'] = HyperoptEstimator(classifier=clf_method, loss_fn=myloss,
                                                                           n_jobs=n_job)
            estim_one_vs_rest[alg.name + '.one_V_all'].fit(X_model_mini, Y_model_mini[:, select_classes],
                                                           valid_size=val_dist, cv_shuffle=False, random_state=seed_val)
            toc_mod = time.time()
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                  "fitting model took %ih:%imin:%is"
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                  % (int((toc_mod - tic_mod) / 60 / 60),
                     int((toc_mod - tic_mod) / 60) - 60 * int((toc_mod - tic_mod) / 60 / 60),
                     int((toc_mod - tic_mod) % 60)))

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                  "scoring on %s" % (alg.name + '.one_V_all'),
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            tic_score = time.time()
            model = estim_one_vs_rest[alg.name + '.one_V_all']
            model_predict = model.predict(X_test_mini)
            algo_scoring[alg.name + '_wF1'] = 1 - myloss(Y_test_mini[:, select_classes], model_predict)
            algo_scoring[alg.name + '_acc'] = 1 - model.score(X_test_mini, Y_test_mini[:, select_classes])
            algo_scoring[alg.name + '_sens'] = recall_score(Y_test_mini[:, select_classes], model_predict,
                                                            average='weighted')
            # algo_scoring[alg.name + '_balanced accu'] = balanced_accuracy_score(Y_test_mini[:, select_classes], model_predict)

            toc_score = time.time()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                  "model score calculations took %ih:%imin:%is"
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                  % (int((toc_score - tic_score) / 60 / 60),
                     int((toc_score - tic_score) / 60) - 60 * int((toc_score - tic_score) / 60 / 60),
                     int((toc_score - tic_score) % 60)))

            # np.save(save_score_path + r"/scores_night_2021_03_13.npy", algo_scoring)

            toc_mod_all = time.time()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                  "fitting models have taken %ih:%imin:%is"
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                  % (int((toc_mod_all - tic_mod_all) / 60 / 60),
                     int((toc_mod_all - tic_mod_all) / 60) - 60 * int((toc_mod_all - tic_mod_all) / 60 / 60),
                     int((toc_mod_all - tic_mod_all) % 60)))

        print("\nmodels trained - next step > model score calculation\n")

        # model - scoring
        tic_score = time.time()

        lars_table = {"model names": [], "wF1": [], "acc": [], "balanced acc": [], "sens": [],
                       "sens-eyem": [], "sens-chew": [], "sens-shiv": [],
                       "sens-elpp": [], "sens-musc": [], "sens-null": [],
                       "acc-eyem": [], "acc-chew": [], "acc-shiv": [], "acc-elpp": [], "acc-musc": [], "acc-null": []}

        algo_scoring1 = dict()
        save_score_path = r'tmp/experiment4'
        # slice_test = int(Y_test.shape[0] / 4)
        # select_alg_new = select_alg
        for alg in [select_alg[args.index]]:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                  "scoring on %s" % (alg.name + '.one_V_all'),
                  "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            model = estim_one_vs_rest[alg.name + '.one_V_all']
            model_predict = model.predict(X_test_mini)
            algo_scoring1[alg.name + '_wF1'] = myloss(Y_test_mini[:, select_classes], model_predict)
            algo_scoring1[alg.name + '_acc'] = 1 - model.score(X_test_mini, Y_test_mini[:, select_classes])
            algo_scoring1[alg.name + '_sens'] = recall_score(Y_test_mini[:, select_classes], model_predict,
                                                             average='weighted')

            # np.save(r"tmp/scores_night_2021_03_22.npy", algo_scoring)
            algo_scoring1[alg.name + '_report'] = classification_report(Y_test_mini[:, select_classes], model_predict,
                                                                        target_names=list(label_column.keys()),
                                                                        zero_division=0, output_dict=True)
            report_catch = algo_scoring1[alg.name + '_report']
            ## TODO: set it to loop mode later in code --- way more pretty
            lars_table["model names"].append(alg.name.split("_")[-1]+'_index%i_subsample_%.2f_run1' % (args.index, args.subsample))
            lars_table["wF1"].append(report_catch["weighted avg"]["f1-score"])
            lars_table["acc"].append(algo_scoring1[alg.name + '_acc'])
            lars_table['balanced acc'].append(report_catch["macro avg"]["recall"])
            lars_table["sens"].append(algo_scoring1[alg.name + '_sens'])
            lars_table['sens-eyem'].append(report_catch[list(label_column.keys())[0]]["recall"])
            lars_table['sens-chew'].append(report_catch[list(label_column.keys())[1]]["recall"])
            lars_table['sens-shiv'].append(report_catch[list(label_column.keys())[2]]["recall"])
            lars_table['sens-elpp'].append(report_catch[list(label_column.keys())[3]]["recall"])
            lars_table['sens-musc'].append(report_catch[list(label_column.keys())[4]]["recall"])
            lars_table['sens-null'].append(report_catch[list(label_column.keys())[5]]["recall"])

            algo_scoring1[alg.name + '_sens-eyem'] = report_catch[list(label_column.keys())[0]]["recall"]
            algo_scoring1[alg.name + '_sens-chew'] = report_catch[list(label_column.keys())[1]]["recall"]
            algo_scoring1[alg.name + '_sens-shiv'] = report_catch[list(label_column.keys())[2]]["recall"]
            algo_scoring1[alg.name + '_sens-elpp'] = report_catch[list(label_column.keys())[3]]["recall"]
            algo_scoring1[alg.name + '_sens-musc'] = report_catch[list(label_column.keys())[4]]["recall"]
            algo_scoring1[alg.name + '_sens-null'] = report_catch[list(label_column.keys())[5]]["recall"]

            algo_scoring1[alg.name + '_acc-eyem'] = accuracy_score(Y_test_mini[:, 0], model_predict[:, 0])
            algo_scoring1[alg.name + '_acc-chew'] = accuracy_score(Y_test_mini[:, 1], model_predict[:, 1])
            algo_scoring1[alg.name + '_acc-shiv'] = accuracy_score(Y_test_mini[:, 2], model_predict[:, 2])
            algo_scoring1[alg.name + '_acc-elpp'] = accuracy_score(Y_test_mini[:, 3], model_predict[:, 3])
            algo_scoring1[alg.name + '_acc-musc'] = accuracy_score(Y_test_mini[:, 4], model_predict[:, 4])
            algo_scoring1[alg.name + '_acc-null'] = accuracy_score(Y_test_mini[:, 5], model_predict[:, 5])
            lars_table['acc-eyem'].append(algo_scoring1[alg.name + '_acc-eyem'])
            lars_table['acc-chew'].append(algo_scoring1[alg.name + '_acc-chew'])
            lars_table['acc-shiv'].append(algo_scoring1[alg.name + '_acc-shiv'])
            lars_table['acc-elpp'].append(algo_scoring1[alg.name + '_acc-elpp'])
            lars_table['acc-musc'].append(algo_scoring1[alg.name + '_acc-musc'])
            lars_table['acc-null'].append(algo_scoring1[alg.name + '_acc-null'])

            # print(classification_report(Y_test_mini[:, select_classes], model_predict,
            #                             target_names=list(label_column.keys()), zero_division=0))

        toc_score = time.time()
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
              "model score calculations took %ih:%imin:%is"
              "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
              % (int((toc_score - tic_score) / 60 / 60),
                 int((toc_score - tic_score) / 60) - 60 * int((toc_score - tic_score) / 60 / 60),
                 int((toc_score - tic_score) % 60)))

        performance_wide = pd.DataFrame.from_dict(lars_table).set_index("model names").round(3)
        performance_tall = pd.DataFrame.from_dict(lars_table, orient='index', columns=lars_table["model names"]).iloc[1:]

        performance_wide.to_csv(save_score_path+r'/ex4_table_%i_run%i_subsample_%.2f.csv' %
                               (args.index, args.var_rerun, args.subsample), encoding='utf-8-sig')
        print("end of code")

        # read_dictionary = np.load(foo + r"/scores.npy", allow_pickle='TRUE').item()

    except:
        print("there was an error - go inspect")
    # estim = HyperoptEstimator(classifier=svc('my_est'), algo=tpe.suggest, ...)

