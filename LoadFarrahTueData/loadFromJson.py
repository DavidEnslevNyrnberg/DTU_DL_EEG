import os, itertools, json, datetime
from collections import defaultdict

# define path to make sure stuff doesn't get saved weird places
os.chdir(os.getcwd())
saveDir = r"C:\Users\anden\PycharmProjects\NovelEEG"+"\\" # ~~~~~~ CHANGE THIS TO YOUR DIR-PATH
farrahData = r"data_farrahtue_EEG\Original participant EEGs"+"\\"
farrahDataDir = saveDir + farrahData

# load json to dict
def loadByJson():
    with open(saveDir+"SampleCode\\recommendedEdfFiles.json", "r") as read_file:
        edfDefDict = json.load(read_file)
    return edfDefDict

