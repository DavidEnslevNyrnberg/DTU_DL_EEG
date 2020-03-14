import os, re, glob, itertools, logging, json
import pandas as pd
from collections import defaultdict

# save files with problems to a .log file
def logwriting(logName, edfFile, path):
    logging.basicConfig(filename=logName, format='%(levelname)-8s %(levelno)d: %(message)s', level=logging.INFO)
    logging.info('Logfile for %s' % edfFile)
    logging.error('\n%s\n\n   Files named %s: %s\n' % (' \n ---\n'.join(path), edfFile, len(path)))
    logging.debug('I HAVE NOT ADDED DEBUG TEXT')
    logging.shutdown()
    return

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG" # ~~~~~~ CHANGE THIS TO YOUR DIR-PATH
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"
farrahDataDir = saveDir + '\\' + farrahData

# flags for code
jsonSave = True

# find all .edf files
pathRootInt = len(saveDir.split('\\'))
edfFoundFiles = [f for f in glob.glob(farrahDataDir + "**/*.edf", recursive=True)]
farrahPaths = ['\\'.join(f.split('\\')[pathRootInt:]) for f in glob.glob(farrahDataDir + "**/*.edf", recursive=True)]
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

# sort into non-complications and complication dicts
edfNonComp = defaultdict(dict)
edfComp = defaultdict(dict)
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


# save paths with non-complications files in json:
if jsonSave:
    with open('recommendedEdfFiles.json', 'w') as fp:
        json.dump(edfNonComp, fp, indent=4)

## TODO: bug after chaning defaultdict{dict{key1: list(), key2: flag, key:data}}
# log bad .edf paths TODO: find a good solution for this
# fileCompli = 0
# fileNCompli = 0
# for file, path in edfDefDict.items():
#     if len(path) == 1:
#         fileNCompli += 1
#         # print('--------------- complication ---------------\n ->', feeNCompli)
#         # print('--------------- non-complication ---------------\n', state, '\n', ' <---\n'.join(cities))
#         # logwriting("myLogFileNice.log", file, path)
#     elif len(path) != 1:
#         fileCompli += 1
#         # print('--------------- complication #', fileCompli, '---------------\n ->',
#         #       '\n', file, '\n ', ' \n ---\n  '.join(path))
#         # print('\nFiles found with name: ', len(path))
#         logwriting("myLogFileComplications.log", file, path)
#     else:
#         print('FAIL -> ', file, ', '.join(path))
