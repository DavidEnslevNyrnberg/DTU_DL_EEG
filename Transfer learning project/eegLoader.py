import os, re, glob, json, sys
import pandas as pd
from collections import defaultdict

# load from json to dict
def jsonLoad(path = False):
    if path is False:
        sys.exit("no path were given to load Json")
    else:
        with open(path, "r") as read_file:
            edfDefDict = json.load(read_file)
    print("\npaths found for loading")
    return edfDefDict

# save a dict to a local .json
def jsonSave(name=False, saveDir=False, dict=False):
    # makes and save dict a
    # t saveDir with name
    with open(saveDir+name, 'w') as fp:
        json.dump(dict, fp, indent=4)
    return print("saving parameters\njson Name: %s\nat location: %s" % (name, saveDir))

# crawls a path for all .edf files
def findEdf(path=False, selectOpt=False, saveDir=False):
    # bypass personal dictionaries
    pathRootInt = len(list(filter(None, saveDir.split('\\'))))
    # find all .edf files in path
    pathList = ['\\'.join(fDir.split('\\')[pathRootInt:]) for fDir in glob.glob(saveDir+path + "**/*.edf", recursive=True)]
    # construct defaultDict for data setting
    edfDict = defaultdict(dict)
    for path in pathList:
        file = path.split('\\')[-1]
        if file in edfDict.keys():
            edfDict[file]["path"].append(path)
            edfDict[file]["deathFlag"] = True
        else:
            edfDict[file]["path"] = []
            edfDict[file]["deathFlag"] = False
            edfDict[file]["path"].append(path)
        edfDict[file]["Files named %s" % file] = len(edfDict[file]["path"])
    return edfDict

def label_load(type=dict()):
    if "BC" in type.keys():
        label_BC
    return list()

def label_BC(ID=int(), edf_name=str()):
    return list()

# reads all labels existing in a time window
def label_TUH(annoPath=False, window=[0,0], saveDir=os.getcwd(), header=None):
    if type(saveDir) is str:
        df = pd.read_csv(saveDir+annoPath, sep=" ", skiprows=1, header=header)
        df.fillna('null', inplace=True)
        within_con0 = (df[0] <= window[0]) & (window[0] <= df[1])
        within_con1 = (df[0] <= window[1]) & (window[1] <= df[1])
        label_TUH = df[df[0].between(window[0], window[1]) |
                       df[1].between(window[0], window[1]) |
                       (within_con0 & within_con1)]
        label_df = label_TUH.rename(columns={0: 't_start', 1: 't_end', 2: 'label', 3: 'confidence'})["label"]
        return_list = label_df.to_numpy().tolist()
    else:
        return_list = saveDir
    return return_list

def label_TUH_full(annoPath=False, window=[0,0], saveDir=os.getcwd(), header=None):
    df = pd.read_csv(saveDir+annoPath, sep=" ", skiprows=1, header=header)
    df.fillna('null', inplace=True)
    within_con0 = (df[0] <= window[0]) & (window[0] <= df[1])
    within_con1 = (df[0] <= window[1]) & (window[1] <= df[1])
    label_TUH = df[df[0].between(window[0], window[1]) |
                   df[1].between(window[0], window[1]) |
                   (within_con0 & within_con1)]
    return label_TUH.rename(columns={0: 't_start', 1: 't_end', 2: 'label', 3: 'confidence'})
