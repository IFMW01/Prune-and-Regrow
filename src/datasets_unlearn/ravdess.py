import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import glob
import librosa
import os
import soundfile as sf
import numpy as np
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm

# NOT PRESENTED IN THESIS

def convert_to_spectograms(data_folder, destination_folder,pipeline=False,downsample=16000):
  os.makedirs(destination_folder, exist_ok=True) 
  max_len = 0
  for index,(path,label) in enumerate(tqdm(data_folder)):
    audio, samplerate = sf.read(path)
    if audio.ndim > 1:
      # Convert to mono 
      audio = audio.mean(axis=1)
    audio = librosa.resample(audio.astype(float),orig_sr=48000,target_sr=downsample)
    audio = torch.tensor(audio).float()
    if max_len < audio.shape[0]:
      max_len = audio.shape[0]

    audio = nn.ConstantPad1d((0, 84351 - audio.shape[0]), 0)(audio)
    if pipeline:
        audio = pipeline(audio)
    label =torch.tensor(label)
    data_dict  = {"feature": audio, "label": label}
    torch.save(data_dict, os.path.join(destination_folder, f"{index}.pth"), )
  

def create_ravdess(pipeline,pipeline_on_wav,dataset_pointer):
    utils.set_seed(42)
    data_folder = f'./ravdess'
    data_path = f'{data_folder}/all_path.npy'
    temp_dir = f'./{pipeline}/{dataset_pointer}'
    # if not os.path.isdir(f'{data_folder}'):
    cv_13 = load_dataset("narad/ravdess", split="train")
    all = np.array([[cv_13[x]['audio']['path'],cv_13[x]['labels']] for x in range(len(cv_13))],dtype=object)
    utils.create_dir(data_folder)
    np.save(data_path, all)

    # if pipeline:
    #     if not os.path.isdir(f'{temp_dir}'):
    utils.create_dir(temp_dir)
    all_data = np.load(data_path,allow_pickle=True)
    convert_to_spectograms(all_data,temp_dir,pipeline_on_wav)
    all_data = glob.glob(f'{temp_dir}/*.pth') 
    train, test = train_test(all_data,pipeline,dataset_pointer,42)
    return train, test

def train_test(all_data,pipeline,dataset_pointer,seed):
  temp_dir = f'./{pipeline}/{dataset_pointer}'
  if os.path.isfile(f'{temp_dir}/train.csv') or os.path.isfile(f'{temp_dir}/test.csv'):
    train = pd.read_csv(f'{temp_dir}/train.csv')
    test = pd.read_csv(f'{temp_dir}/test.csv')
    train = (train.values.flatten().tolist())
    test = (test.values.flatten().tolist())
  else:
    train, test = train_test_split(all_data, test_size=0.2, random_state=seed,shuffle=True)
    train_path = f'{temp_dir}/train.csv'
    test_path = f'{temp_dir}/test.csv'
    pd.DataFrame(train).to_csv(f'{train_path}', index=False)
    pd.DataFrame(test).to_csv(f'{test_path}', index=False)
    
  return train, test    

