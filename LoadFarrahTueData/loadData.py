import os, re, glob, itertools, logging, json
import pandas as pd
from collections import defaultdict

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\" # ~~~~~~ CHANGE THIS TO YOUR DIR-PATH
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"
farrahDataDir = saveDir + farrahData

# flags for code
jsonSave = False

# find all .edf files
pathRootInt = -5
edfFoundFiles = [f for f in glob.glob(farrahDataDir + "**/*.edf", recursive=True)]
farrahPaths = ['\\'.join(f.split('\\')[pathRootInt:]) for f in glob.glob(farrahDataDir + "**/*.edf", recursive=True)]
# construct defaultDict for data setting
edfDefDict = defaultdict(dict)
for path in farrahPaths:
    file = path.split('\\')[-1]
    edfDefDict[file]["path"] = []
    edfDefDict[file]["deathFlag"] = False
for path in farrahPaths:
    file = path.split('\\')[-1]
    edfDefDict[file]["path"].append(saveDir + path)
    if len(edfDefDict[file]["path"]) != 1:
        edfDefDict[file]["deathFlag"] = True

# sort into non-complications and complication dicts
edfNonComp = edfDefDict.copy()
edfComp = edfDefDict.copy()
deathIDs = [ID for ID in edfNonComp.keys() if edfNonComp[ID]["deathFlag"] == True]
liveIDs = [ID for ID in edfComp.keys() if edfComp[ID]["deathFlag"] == False]
for rmID in deathIDs:
    edfNonComp.pop(rmID)
for rmID in liveIDs:
    edfComp.pop(rmID)

# save paths with non-complications files in json:
if jsonSave:
    with open('data.json', 'w') as fp:
        json.dump(edfNonComp, fp, indent=4)

