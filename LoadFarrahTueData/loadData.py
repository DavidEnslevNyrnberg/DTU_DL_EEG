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


# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\"  # ~~~ What is your execute path?
# farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"  # ~~~ What is the name of your data folder?
farrahData = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset\01_tcp_ar\100\00010023\s002_2013_02_21"
jsonDir = r"edfFiles.json"  # ~~~ Where is your json folder?
farrahDataDir = saveDir + '\\' + farrahData

# flags for saving TODO: make if-statements as functions in the class
jsonSave = False
debugLog = False

# find all .edf files
pathRootInt = len(saveDir.split('\\'))
farrahPaths = ['\\'.join(fDir.split('\\')[pathRootInt:]) for fDir in glob.glob(farrahDataDir + "**/*.edf", recursive=True)]
# construct defaultDict for data setting
edfDefDict = defaultdict(dict)
for path in farrahPaths:
    file = path.split('\\')[-1]
    if file in edfDefDict.keys():
        edfDefDict[file]["path"].append(path)
        edfDefDict[file]["deathFlag"] = True
    else:
        edfDefDict[file]["path"] = []
        edfDefDict[file]["deathFlag"] = False
        edfDefDict[file]["path"].append(path)
    edfDefDict[file]["Files named %s" % file] = len(edfDefDict[file]["path"])

# sort into non-complications and complication dicts TODO: might update this to something smarter
edfNonComp = {ID: v for (ID, v) in edfDefDict.items() if edfDefDict[ID]["deathFlag"] is False}
edfComp = {ID: v for (ID, v) in edfDefDict.items() if edfDefDict[ID]["deathFlag"] is True}

# join with Tue/Farrah annotations
# xlsxName = 'NEW MGH File Annotations.xlsx'
# xlsxAnnotations = pd.read_excel(farrahDataDir+xlsxName, sheet_name=None)
#
# xlsxAnnotations0 = pd.read_excel(farrahDataDir+xlsxName, sheet_name=0)
# print(xlsxAnnotations0.columns)
# xlsxAnnotations2 = pd.read_excel(farrahDataDir+xlsxName, sheet_name=2)
# print(xlsxAnnotations2.columns)
# xlsxAnnotations3 = pd.read_excel(farrahDataDir+xlsxName, sheet_name=3)
# print(xlsxAnnotations3.columns)

# save paths with non-complications files in json
if jsonSave:
    with open(saveDir+'edfFiles.json', 'w') as fp:
        json.dump(edfNonComp, fp, indent=4)

# .edf paths in a non-comp and comp json as logging file
if debugLog is True:
    with open(saveDir+'nCompEdf.json', 'w') as fp:
        json.dump(edfNonComp, fp, indent=4)
    with open(saveDir+'compEdf.json', 'w') as fp:
        json.dump(edfComp, fp, indent=4)

# TODO: functions in this class
# load by finding .edf files in a path
# load by .json (CHECK)
# save into .json
# load annotations
# select by flags
# load Tensor