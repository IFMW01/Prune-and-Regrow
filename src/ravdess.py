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


def convert_to_spectograms(data_folder, destination_folder,pipeline=False,downsample=16000):
  os.makedirs(destination_folder, exist_ok=True) 
  for index,(path,wav,label) in enumerate(tqdm(data_folder)):
    audio, samplerate = sf.read(wav)
    audio = librosa.resample(audio.astype(float),orig_sr=samplerate,target_sr=downsample)
    audio = torch.tensor(audio).float()
    audio = nn.ConstantPad1d((0, downsample - audio.shape[0]), 0)(audio)
    if pipeline:
        audio = pipeline(audio)
    label =torch.tensor(label)
    data_dict  = {"feature": audio, "label": label}
    torch.save(data_dict, os.path.join(destination_folder, f"{index}.pth"), )

def create_ravdess(pipeline,pipeline_on_wav,dataset_pointer):
    utils.set_seed(42)
    data_folder = f'./ravdess'
    data_path = f'{data_folder}/all_path.json'
    temp_dir = f'./{pipeline}/{dataset_pointer}'
    if not os.path.isdir(f'{data_folder}'):
        cv_13 = load_dataset("narad/ravdess", split="train")
        all =  [[cv_13[x]['audio']['path'],cv_13[x]['audio']['array'],cv_13[x]['labels']] for x in range(len(cv_13))]
        utils.create_dir(data_folder)
        all_path = f'{data_folder}/all_data.json'
        with open(f'{data_path}', 'w') as f:
            json.dump(all, f)
    if pipeline:
        if not os.path.isdir(f'{temp_dir}'):
            utils.create_dir(temp_dir)
            with open('my_list.json', 'r') as f:
                all_data = json.load(f)
            convert_to_spectograms(all_data,temp_dir,pipeline_on_wav)
        all_data = glob.glob(f'{temp_dir}/*.pth') 
    
    return all_data

def train_test(all_data,pipeline,dataset_pointer,seed):
  temp_dir = f'./{pipeline}/{dataset_pointer}'
  if os.path.isfile(f'{temp_dir}/train.csv') or os.path.isfile(f'{temp_dir}/test.csv'):
    train = pd.read_csv(f'{temp_dir}/train.csv')
    test = pd.read_csv(f'{temp_dir}/test.csv')
    train = (train.values.flatten().tolist())
    test = (test.values.flatten().tolist())
  else:
    train, test = train_test_split(all_data, test_size=0.2, random_state=seed,shuffle=True)
    train_path = f'./{temp_dir}/train.csv'
    test_path = f'./{temp_dir}/test.csv'
    pd.DataFrame(train).to_csv(f'{train_path}', index=False)
    pd.DataFrame(test).to_csv(f'{test_path}', index=False)
  
  return train, test    

def load_mia_dataset(pipeline,pipeline_on_wav,dataset_pointer):
    data_folder = './AudioMNIST/data/*/'
    temp_dir = f'./{pipeline}/{dataset_pointer}'
    if os.path.isdir(f'{temp_dir}'):
        all_data = glob.glob(f'{temp_dir}/*.pth')   
    else:
      if not os.path.isdir('./AudioMNIST'):
          git_clone_command = ['git', 'clone', 'https://github.com/soerenab/AudioMNIST.git']
          subprocess.run(git_clone_command, check=True)
      all_data = glob.glob(f'{data_folder}*.wav')
      if pipeline:
          convert_to_spectograms(all_data,temp_dir,pipeline_on_wav)
          all_data = glob.glob(f'{temp_dir}/*.pth')   
      repository_path = './AudioMNIST'
      shutil.rmtree(repository_path)
    
    return all_data   