import glob
import librosa
import os
import torch
import soundfile as sf
import torch.nn as nn
import numpy as np
import subprocess
import shutil
import random
import json
import datasets
import utils
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS


def convert_to_spectograms(data_folder, destination_folder,pipeline=False,downsample=16000):
  os.makedirs(destination_folder, exist_ok=True) 
  max_len = 0
  for index,(audio,label) in enumerate(tqdm(data_folder)):
    if audio.ndim > 1:
      # Convert to mono 
      audio = audio.mean(axis=1)
    audio = librosa.resample(audio.astype(float),orig_sr=16000,target_sr=downsample)
    audio = torch.tensor(audio).float()
    if max_len < audio.shape[0]:
      max_len = audio.shape[0]

    audio = nn.ConstantPad1d((0, 16000 - audio.shape[0]), 0)(audio)
    if pipeline:
        audio = pipeline(audio)
    label =torch.tensor(label)
    data_dict  = {"feature": audio, "label": label}
    torch.save(data_dict, os.path.join(destination_folder, f"{index}.pth"), )
  

def create_speechcommands(pipeline,pipeline_on_wav,dataset_pointer):
    utils.set_seed(42)
    train_temp_dir = f'./{pipeline}/{dataset_pointer}/Train'
    test_temp_dir = f'./{pipeline}/{dataset_pointer}/Train'
    if not os.path.isdir(f'{train_temp_dir}'):
      train_list = SubsetSC("training") 
      test_list = SubsetSC("training") 
      train_path_arr = []
      test_path_arr = []
      with open("./SpeechCommands/speech_commands_v0.02/training_list.txt", "r") as file:
        for line in file:
            train_path_arr.append((line.strip()))

      with open("./SpeechCommands/speech_commands_v0.02/testing_list.txt", "r") as file:
        for line in file:
            test_path_arr.append((line.strip()))
      
      sc_train = []
      sc_test = []
      if pipeline:
        utils.create_dir(train_temp_dir)
        utils.create_dir(test_temp_dir)
        for i in range(len(train_list)):
          sc_train.append((train_path_arr[i],train_list[i][4]))
        for i in range(len(test_list)):
          sc_train.append((test_path_arr[i],test_list[i][4]))
        convert_to_spectograms(sc_train,train_temp_dir,pipeline_on_wav)
        convert_to_spectograms(sc_test,test_temp_dir,pipeline_on_wav)

    train_set = glob.glob(f'{train_temp_dir}/*.pth') 
    test_set = glob.glob(f'{test_temp_dir}/*.pth') 
    return train_set,test_set

def train_test(all_data,pipeline,dataset_pointer,seed):
  temp_dir = f'./{pipeline}/{dataset_pointer}'
  # if os.path.isfile(f'{temp_dir}/train.csv') or os.path.isfile(f'{temp_dir}/test.csv'):
  #   train = pd.read_csv(f'{temp_dir}/train.csv')
  #   test = pd.read_csv(f'{temp_dir}/test.csv')
  #   train = (train.values.flatten().tolist())
  #   test = (test.values.flatten().tolist())
  # else:
  train, test = train_test_split(all_data, test_size=0.2, random_state=seed,shuffle=True)
  train_path = f'{temp_dir}/train.csv'
  test_path = f'{temp_dir}/test.csv'
  pd.DataFrame(train).to_csv(f'{train_path}', index=False)
  pd.DataFrame(test).to_csv(f'{test_path}', index=False)
  
  return train, test    

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self,subset: str = None):
        super().__init__("./", download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            filepath = os.path.join(self._path, 'training_list.txt')
            self._walker = [w for w in self._walker if w not in excludes]
            with open(filepath, "w") as file:
                for item in self._walker:
                    file.write(str(item) + "\n")

        elif subset == "all":
            self._walker = [w for w in self._walker]

