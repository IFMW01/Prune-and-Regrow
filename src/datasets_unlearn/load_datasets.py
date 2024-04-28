import torchaudio
import os
import torch
import librosa
import numpy as np
import torch
import torchvision
import random
import torchvision.transforms as transforms
from datasets_unlearn import ravdess
from datasets_unlearn  import audioMNIST
from datasets_unlearn  import speech_commands
import utils

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

seed = 42
labels = np.load('./labels/speech_commands_labels.npy')
labels = labels.tolist()

def load_datasets(dataset_pointer :str,pipeline:str,unlearnng:bool):
    global labels
    if pipeline == 'mel':
        pipeline_on_wav = WavToMel()
    elif pipeline =='spec':
        pipeline_on_wav = WavToSpec()
    if not os.path.exists(dataset_pointer):
            print(f"Downloading: {dataset_pointer}")
    if dataset_pointer == 'SpeechCommands':
        train_set,test_set = speech_commands.create_speechcommands(pipeline,pipeline_on_wav,dataset_pointer)
        labels = np.load('./labels/speech_commands_labels.npy')
    elif dataset_pointer == 'audioMNIST':
        train_set, test_set = audioMNIST.create_audioMNIST(pipeline,pipeline_on_wav,dataset_pointer)
        labels = np.load('./labels/audiomnist_labels.npy')
    elif dataset_pointer == 'Ravdess':
        train_set, test_set = ravdess.create_ravdess(pipeline,pipeline_on_wav,dataset_pointer)
        labels = np.load('./labels/ravdess_label.npy')
    elif dataset_pointer == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    labels = labels.tolist()

    if unlearnng:
        return train_set,test_set
    device  = utils.get_device()
    if dataset_pointer == 'SpeechCommands' or dataset_pointer == 'audioMNIST' or dataset_pointer == 'Ravdess':
        train_set = DatasetProcessor(train_set,device)
        test_set = DatasetProcessor(test_set,device)

    train_loader = DataLoader(train_set, batch_size=256,shuffle=True)
    train_eval_loader = DataLoader(train_set, batch_size=256,shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256,shuffle=False)
        
    return train_loader,train_eval_loader,test_loader

class DatasetProcessor(Dataset):
  def __init__(self, annotations, device):
    self.audio_files = annotations
    self.features = [] 
    self.labels = [] 
    for idx, path in enumerate(self.audio_files):
       d = torch.load(path)
       d["feature"] = d["feature"][None,:,:]
       self.features.append(d["feature"].to(device))
       self.labels.append(d["label"].to(device))
    # self.features = torch.tensor(self.features)
    # self.labels = torch.tensor(self.labels)

  def __len__(self):
    return len(self.audio_files)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

class DatasetProcessor_randl(Dataset):
  def __init__(self, annotations,device,num_classes):
    self.audio_files = annotations
    self.features = [] #torch.zeros(size=(len(self.audio_files),))
    self.labels = [] #torch.zeros(size=(len(self.audio_files),))
    for idx, path in enumerate(self.audio_files):
       d = torch.load(path)
       d["feature"] = d["feature"][None,:,:]
       self.features.append(d["feature"].to(device))
       new_label = d["label"] 
       while new_label == d["label"]:
            new_label = random.randint(0, num_classes)
       new_label = torch.tensor(new_label, dtype=torch.int8)
       d["label"] = new_label
       self.labels.append(d["label"].to(device))


  def __len__(self):
    return len(self.audio_files)
  
  def __getitem__(self, idx):
    """Get the item at idx and apply the transforms."""
    # audio_path = self.audio_files[idx]
    # data = torch.load(audio_path)
    # data["feature"] = data["feature"][None,:,:]
    # new_label = data["label"] 
    # while new_label == data["label"]:
    #   new_label = random.randint(0, (len(labels)-1))
    # torch.tensor(new_label, dtype=torch.int8)
    # data["label"] = new_label
    # return data["feature"], data["label"]   
    return self.features[idx], self.labels[idx] 

class WavToMel(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        n_fft=512,
        n_mel=32
    ):
        super().__init__()

        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mel, sample_rate=input_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform)

        mel = self.mel_scale(spec)

        return mel
    
class WavToSpec(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        n_fft=512,
        n_mel=32
    ):
        super().__init__()

        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mel, sample_rate=input_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform)
        spec = torch.from_numpy(librosa.power_to_db(spec))
        return spec

