import os, json, sys
from collections import defaultdict

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\" # ~~~~~~ CHANGE THIS TO YOUR DIR-PATH
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"
jsonDir = r"edfFiles.json"
farrahDataDir = saveDir + farrahData

# load from json to dict
def jsonLoad(path = False):
    if path is False:
        sys.exit("no path were given to load Json")
    else:
        with open(path, "r") as read_file:
            edfDefDict = json.load(read_file)
    print("\npaths found for loading")
    return edfDefDict
