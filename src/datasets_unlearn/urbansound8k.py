import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import glob
import librosa
import requests
import os
import soundfile as sf
import shutil
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import tarfile
import os
import shutil


def convert_data(dataset, destination_folder,pipeline=False,downsample=16000):
  os.makedirs(destination_folder, exist_ok=True)
  for index,row in tqdm(dataset.iterrows()):
    path = row['filepath']
    file_name = (path.split('/')[-1]).split('.')[0]
    if not os.path.isfile(f'./{path}'):
      continue
    audio, samplerate = sf.read(row['filepath'])
    if audio.ndim > 1:
      # Convert to mono
      audio = audio.mean(axis=1)
    if samplerate != downsample:
      audio = librosa.resample(audio.astype(float),orig_sr=samplerate,target_sr=downsample)
    audio = torch.tensor(audio).float()
    audio = nn.ConstantPad1d((0, downsample - audio.shape[0]), 0)(audio)
    if pipeline:
      audio = pipeline(audio)
    label = torch.tensor(row['classID'])
    data_dict  = {"feature": audio, "label": label}
    torch.save(data_dict, os.path.join(destination_folder, f"{file_name}.pth"), )


def create_UrbanSound8K(pipeline,pipeline_on_wav,dataset_pointer):
    data_folder = './UrbanSound8K'
    temp_dir = f'./{pipeline}/{dataset_pointer}'
    if os.path.isdir(f'{temp_dir}'):
        all_data = glob.glob(f'{temp_dir}/*.pth')
    else:
      # Saves the loaded data in the correct waveform format in a separate directory
      if not os.path.isdir(data_folder):
        url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
        output_file = "urban8k.tgz"
        response = requests.get(url, stream=True)
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        with tarfile.open(output_file, "r:gz") as tar:
            tar.extractall()
        os.remove(output_file)
        os.makedirs(temp_dir, exist_ok=True)
        shutil.copy('./UrbanSound8K/metadata/UrbanSound8K.csv', f'{temp_dir}/UrbanSound8K.csv')


    if not os.path.isfile(f'{temp_dir}/train.csv') or not os.path.isfile(f'{temp_dir}/test.csv'):
          dataset = pd.read_csv(f'{temp_dir}/UrbanSound8K.csv')
          filepaths = []
          for i, row in dataset.iterrows():
              filepaths.append(os.path.join('UrbanSound8K/audio', 'fold'+str(row['fold']), row['slice_file_name']))
          dataset['filepath'] = filepaths
          convert_data(dataset,temp_dir,pipeline_on_wav)
          shutil.rmtree(data_folder)
    all_data = glob.glob(f'{temp_dir}/*.pth')
    train, test = train_test(all_data,temp_dir,42)
    return train, test

def train_test(all_data,temp_dir,seed):
  if os.path.isfile(f'{temp_dir}/train.csv') or os.path.isfile(f'{temp_dir}/test.csv'):
    train = pd.read_csv(f'{temp_dir}/train.csv')
    test = pd.read_csv(f'{temp_dir}/test.csv')
    train = (train.values.flatten().tolist())
    test = (test.values.flatten().tolist())
  else:
    train, test = train_test_split(all_data, test_size=0.2, random_state=seed,shuffle=True)
    print("VERIFYING UrbanSound8K DATASET")
    train_path = f'{temp_dir}/train.csv'
    test_path = f'{temp_dir}/test.csv'
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
  return train, test
