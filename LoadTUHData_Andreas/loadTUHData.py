
import os, mne, re, glob
import numpy as np
import matplotlib.pyplot as plt

class Load_TUH:
    def __init__(self):
        pass

    def make_dict(self,dataset_path=r"C:\Users\Andreas\Desktop\KID\Fagproject\Data\artifact_dataset"):
        """
        makes a dictionary over all 
        """
        EEG_id=0
        EEG_dict={}
        for ref in  os.listdir(dataset_path): #should loop over reference types 01_tcp_ar (avage reference) or 02_tcp_le (link era) or 03_tcp_ar_a(?)            
            if re.search("txt",ref)==None: #Filter out txt files
                ref_path=os.path.join(dataset_path,ref)
                for id in os.listdir(ref_path):
                    id_path=os.path.join(ref_path,id)
                    for patient in os.listdir(id_path):
                        patient_path=os.path.join(id_path,patient)
                        for session in os.listdir(patient_path):
                            session_path=os.path.join(patient_path,session)
                            EEG_dict.update({EEG_id:{"ref":ref,"id":id,"patient_id":patient,"session":session,"path":session_path}})
                            EEG_id+=1
        self.EEG_dict=EEG_dict




    def load_id_one(self,id):
        """
        Loade one spectrogram with patient id.
        id sting 3 numbers eg. 001:
        ref: 01_tcp_ar (avage reference) or 02_tcp_le (link era) or 03_tcp_ar_a(?)
        """

        file_folder=self.EEG_dict[id]["path"]
        file_path=glob.glob(file_folder + "**/*.edf", recursive=False)
        EEG_raw=mne.io.read_raw_edf(file_path[0],preload=True)
        
        EEG_raw=sort(EEG=EEG_raw,montage='BC',rename=False)

        return EEG_raw

def sort(EEG,montage,rename=False):
    """
    EEG mne raw data.
    montage str '10-20' or "BC"
    rename bool replace rename channel names if true
    Return mne data 
    """

    bool_index=np.zeros(len(EEG.ch_names))
    if montage=="10-20":
        setup=ten_twenty()
    elif montage=="BC":
        setup=BC()
    else:
        raise NameError('Invalid montage')

    for channel in setup:
        index=mne.pick_channels_regexp(EEG.ch_names,regexp="EEG "+channel) #May need extra debugging
        
        if index==[]:
            #IF someone want to gennerate missing channels thoug interpolation this is the place.
            raise NameError(f"Channel {channel} is not in dataset")

        index=index[0]
        if rename==True:
            EEG.ch_names[index]=channel
        
        bool_index[index]=True
    EEG_new=EEG.drop_channels(np.array(EEG.ch_names)[bool_index==False])
    return EEG_new

def ten_twenty():
    '''
    Return af standard 10-20 setup. 
    Note: does not return T1 , T2 (becourse i don't know what they are Andreas Madsen).         
    '''
    return ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ','A1','A2']

def BC():
    """
    Return brain capture setup channels
    """
    return ['F3','F4','C3','C4','P3','P4','O1','O2','FZ','CZ','PZ','A1','A2']
c=Load_TUH()
c.make_dict()
EEG=c.load_id_one(1)
print(EEG.ch_names)