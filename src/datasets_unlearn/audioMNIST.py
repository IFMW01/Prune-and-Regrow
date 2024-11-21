import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import glob
import librosa
import os
import soundfile as sf
import numpy as np
import subprocess
import shutil
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

labels = np.load('./labels/audiomnist_labels.npy')
labels = labels.tolist()

#Script to process audioMNIST

def convert_to_spectograms(data_folder, destination_folder,pipeline=False,downsample=16000):
  os.makedirs(destination_folder, exist_ok=True) 
  for idx, data in enumerate(tqdm(data_folder)):
    audio, samplerate = sf.read(data)
    audio = librosa.resample(audio.astype(float),orig_sr=samplerate,target_sr=downsample)
    audio = torch.tensor(audio).float()
    audio = nn.ConstantPad1d((0, downsample - audio.shape[0]), 0)(audio)
    if pipeline:
        audio = pipeline(audio)
    label =torch.tensor(int(os.path.basename(data)[0]))
    data_dict  = {"feature": audio, "label": label}
    torch.save(data_dict, os.path.join(destination_folder, f"{idx}.pth"), )
      

def create_audioMNIST(pipeline,pipeline_on_wav,dataset_pointer):
    data_folder = './AudioMNIST/data/*/'
    temp_dir = f'./{pipeline}/{dataset_pointer}'
    if os.path.isdir(f'{temp_dir}'):
        all_data = glob.glob(f'{temp_dir}/*.pth')   
    else:
      # Saves the loaded data in the correct waveform format in a separate directory
      if not os.path.isdir('./AudioMNIST'):
          git_clone_command = ['git', 'clone', 'https://github.com/soerenab/AudioMNIST.git']
          subprocess.run(git_clone_command, check=True)
      all_data = glob.glob(f'{data_folder}*.wav')
      if pipeline:
          convert_to_spectograms(all_data,temp_dir,pipeline_on_wav)
          all_data = glob.glob(f'{temp_dir}/*.pth')   
      repository_path = './AudioMNIST'
      shutil.rmtree(repository_path)
    train, test = train_test(all_data,pipeline,dataset_pointer,42)
    return train, test
    
def train_test(all_data,pipeline,dataset_pointer,seed):
  temp_dir = f'./{pipeline}'
  if os.path.isfile(f'{temp_dir}/train.csv') or os.path.isfile(f'{temp_dir}/test.csv'):
    train = pd.read_csv(f'{temp_dir}/train.csv')
    test = pd.read_csv(f'{temp_dir}/test.csv')
    train = (train.values.flatten().tolist())
    test = (test.values.flatten().tolist())
  else:
    train, test = train_test_split(all_data, test_size=0.2, random_state=seed,shuffle=True)
    train_path = f'./{pipeline}/train.csv'
    test_path = f'./{pipeline}/test.csv'
    pd.DataFrame(train).to_csv(f'{train_path}', index=False)
    pd.DataFrame(test).to_csv(f'{test_path}', index=False)
  
  return train, test


