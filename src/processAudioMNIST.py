import subprocess
import os
import glob
import random
import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import utils
import shutil
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from joblib import dump
from joblib import load
seed = 42
random.seed(seed)

def write_metadata(filepath,metadata):
  with open(f'{filepath}', 'w', newline='') as file:
    for path in metadata:
        file.write("%s\n" % list)

def audioMNIST():
    utils.set_seed(seed)
    git_clone_command = ['git', 'clone', 'git clone https://github.com/soerenab/AudioMNIST.git']
    subprocess.run(git_clone_command, check=True)
    with open('./AudioMNIST/data/audioMNIST_meta.txt', 'r') as file:
        dict_str = file.read()
        dictionary = eval(dict_str)

    root_directory = './AudioMNIST/data/'
    dataset = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        wav_files = glob.glob(os.path.join(dirpath, '*.wav'))
        
        if wav_files:
            for wav_file in wav_files:
                file_path = os.path.basename(wav_file)
                last_part = file_path.split("_")
                label = last_part[0]
                speaker_id = last_part[1]
                gender = dictionary[f'{speaker_id}']['gender']
                if gender == 'male':
                    gender = 0
                else:
                    gender = 1
                samplerate, data = wavfile.read(str(wav_file))
                data = librosa.resample(data.astype(float),orig_sr=samplerate,target_sr=16000)
                dataset.append([data,file_path,speaker_id,gender,label])
                del data
    repository_path = './AudioMNIST'
    shutil.rmtree(repository_path)
    train_set, test_set = train_test_split(dataset,random_state=seed, test_size=0.20,shuffle=True)
    return train_set,test_set
