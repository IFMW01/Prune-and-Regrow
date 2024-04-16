import glob
import librosa
import os
import torch
import soundfile as sf
import utils
import torch.nn as nn
import subprocess
import shutil
import random
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def audio2spec(audio):
  return audio

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
    utils.set_seed(42)
    data_folder = './AudioMNIST/data/*/'
    temp_dir = f'./{pipeline}/{dataset_pointer}'
    all_data = glob.glob(f'{data_folder}*.wav')
    if not os.path.isdir('./AudioMNIST'):
        git_clone_command = ['git', 'clone', 'https://github.com/soerenab/AudioMNIST.git']
        subprocess.run(git_clone_command, check=True)
    
    if pipeline:
        convert_to_spectograms(all_data,temp_dir,pipeline_on_wav)
        all_data = glob.glob(f'{temp_dir}/*.pth') 
    
    # repository_path = './AudioMNIST'
    # shutil.rmtree(repository_path)
    
    return all_data
    
def train_test(all_data,pipeline,dataset_pointer,seed):
       train, test = train_test_split(all_data, test_size=0.2, random_state=seed)
       train_path = f'./{pipeline}/{dataset_pointer}/train.csv'
       test_path = f'./{pipeline}/{dataset_pointer}/test.csv'
       pd.DataFrame(train).to_csv(f'{train_path}', index=False)
       pd.DataFrame(test).to_csv(f'{test_path}', index=False)
       return train, test

class AudioMNISTDataset(Dataset):
  def __init__(self, annotations):
    self.audio_files = annotations

  def __len__(self):
    return len(self.audio_files)
  
  def __getitem__(self, idx):
    """Get the item at idx and apply the transforms."""
    audio_path = self.audio_files[idx]
    data = torch.load(audio_path)
    data["feature"] = data["feature"][None,:,:]
    return data["feature"], data["label"]

# def audioMNIST_train_test():
#     dataset = load()
#     train_set, test_set = train_test_split(dataset,random_state=seed, test_size=0.20,shuffle=True)
#     print(type(train_set))
#     print(train_set[0])
#     # train_set = torch.tensor(tuple(train_set)) 
#     # test_set = torch.tensor(tuple(test_set))  
#     return train_set,test_set
    
# def audioMNIST_all():
#     dataset= load()
#     random.shuffle(dataset)
#     dataset = torch.as_tensor(dataset)  
#     return dataset

# def load():
#     utils.set_seed(seed)

#     with open('./AudioMNIST/data/audioMNIST_meta.txt', 'r') as file:
#         dict_str = file.read()
#         dictionary = eval(dict_str)

#     root_directory = './AudioMNIST/data/'
#     dataset = []
#     for dirpath, dirnames, filenames in os.walk(root_directory):
#         wav_files = glob.glob(os.path.join(dirpath, '*.wav'))
        
#         if wav_files:
#             for wav_file in wav_files:
#                 file_path = os.path.basename(wav_file)
#                 print(file_path)
#                 last_part = file_path.split("_")
#                 print(last_part)
#                 label = int(last_part[0][-1])
#                 print(label)
#                 break
#                 speaker_id = last_part[1]
#                 gender = dictionary[f'{speaker_id}']['gender']
#                 if gender == 'male':
#                     gender = 0
#                 else:
#                     gender = 1
#                 samplerate, data = wavfile.read(str(wav_file))
#                 data = librosa.resample(data.astype(float),orig_sr=samplerate,target_sr=16000)
#                 # data = torch.tensor(data)
#                 dataset.append((data,int(speaker_id),gender,label))
#                 del data
#     repository_path = './AudioMNIST'
#     shutil.rmtree(repository_path)
#     return tuple(dataset)

# def write_metadata(filepath,metadata):
#   with open(f'{filepath}', 'w', newline='') as file:
#     for path in metadata:
#         file.write("%s\n" % list)

