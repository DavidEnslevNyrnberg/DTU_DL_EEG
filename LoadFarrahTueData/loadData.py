import os, re, glob, json, sys
import pandas as pd
from collections import defaultdict

#### TODO: make all this a class-object

# load from json to dict
def jsonLoad(path = False):
    if path is False:
        sys.exit("no path were given to load Json")
    else:
        with open(path, "r") as read_file:
            edfDefDict = json.load(read_file)
    print("\npaths found for loading")
    return edfDefDict

def jsonSave(name=False, saveDir=False, dict=False):
    # makes and save dict at saveDir with name
    with open(saveDir+name, 'w') as fp:
        json.dump(dict, fp, indent=4)
    return print("saving parameters\njson Name: %s\nat location: %s" % (name, saveDir))

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

def annoTUH(annoPath=False, window=[0,0], saveDir=os.getcwd(), header=None):
    df = pd.read_csv(saveDir+annoPath, sep=" ", skiprows=1, header=header)
    df.fillna('null', inplace=True)
    annoTUH = df[df[0].between(window[0], window[1]) | df[1].between(window[0], window[1])]
    return annoTUH

# # define path to make sure stuff doesn't get saved weird places
# os.chdir(os.getcwd())
# saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\"  # ~~~ What is your execute path?
# farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"  # ~~~ What is the name of your data folder?
# tuhData = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\**\01_tcp_ar"+"\\" #\100\00010023\s002_2013_02_21
# jsonDir = r"edfFiles2.json"  # ~~~ Where is your json folder?
# farrahDataDir = saveDir + farrahData
# tuhDataDir = saveDir + tuhData
# # edfDefDict["00010023_s002_t005.edf"]
# # flags for saving TODO: make if-statements as functions in the class
# jsonSave = False
# debugLog = False
#
# edfTUH = findEdf(path=tuhData, saveDir=saveDir)
# # find all .edf files
# pathRootInt = len(list(filter(None, saveDir.split('\\'))))
# farrahPaths = ['\\'.join(fDir.split('\\')[pathRootInt:]) for fDir in glob.glob(farrahDataDir + "**/*.edf", recursive=True)]
# tuhPaths = ['\\'.join(fDir.split('\\')[pathRootInt:]) for fDir in glob.glob(tuhDataDir + "**/01_tcp_ar/**/*.edf", recursive=True)]
# # construct defaultDict for data setting
# edfDefDict = defaultdict(dict)
# for path in tuhPaths:
#     file = path.split('\\')[-1]
#     if file in edfDefDict.keys():
#         edfDefDict[file]["path"].append(path)
#         edfDefDict[file]["deathFlag"] = True
#     else:
#         edfDefDict[file]["path"] = []
#         edfDefDict[file]["deathFlag"] = False
#         edfDefDict[file]["path"].append(path)
#     edfDefDict[file]["Files named %s" % file] = len(edfDefDict[file]["path"])
#
# # sort into non-complications and complication dicts TODO: might update this to something smarter
# # edfNonComp = {ID: v for (ID, v) in edfDefDict.items() if edfDefDict[ID]["deathFlag"] is False}
# # edfComp = {ID: v for (ID, v) in edfDefDict.items() if edfDefDict[ID]["deathFlag"] is True}
#
# print('pause')
#
# # join with Tue/Farrah annotations
# # xlsxName = 'NEW MGH File Annotations.xlsx'
# # xlsxAnnotations = pd.read_excel(farrahDataDir+xlsxName, sheet_name=None)
# #
# # xlsxAnnotations0 = pd.read_excel(farrahDataDir+xlsxName, sheet_name=0)
# # print(xlsxAnnotations0.columns)
# # xlsxAnnotations2 = pd.read_excel(farrahDataDir+xlsxName, sheet_name=2)
# # print(xlsxAnnotations2.columns)
# # xlsxAnnotations3 = pd.read_excel(farrahDataDir+xlsxName, sheet_name=3)
# # print(xlsxAnnotations3.columns)
#
# # TODO: functions in this class
# # load by finding .edf files in a path
# # load by .json (CHECK)
# # save into .json
# # load annotations
# # select by flags
# # load Tensor
