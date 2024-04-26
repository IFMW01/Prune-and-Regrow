import torchaudio
import os
import preprocess as pp
import torch
import librosa
import numpy as np
import torch
import torchvision
import random
import torchvision.transforms as transforms
import ravdess
import audioMNIST
import speech_commands
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
        print(train_set[0])
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
    
    if dataset_pointer == 'SpeechCommands' or dataset_pointer == 'audioMNIST' or dataset_pointer == 'Ravdess':
        train_set = DatasetProcessor(train_set)
        print(train_set[0])
        print(train_set[0][0].shape)
        test_set = DatasetProcessor(test_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                            shuffle=True, num_workers=2)
    train_eval_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                            shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                            shuffle=False, num_workers=2)
        
    return train_loader,train_eval_loader,test_loader

class DatasetProcessor(Dataset):
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

class DatasetProcessor_randl(Dataset):
  def __init__(self, annotations):
    self.audio_files = annotations

  def __len__(self):
    return len(self.audio_files)
  
  def __getitem__(self, idx):
    """Get the item at idx and apply the transforms."""
    audio_path = self.audio_files[idx]
    data = torch.load(audio_path)
    data["feature"] = data["feature"][None,:,:]
    new_label = data["label"] 
    while new_label == data["label"]:
      new_label = random.randint(0, (len(labels)-1))
    torch.tensor(new_label, dtype=torch.int8)
    data["label"] = new_label
    return data["feature"], data["label"]    

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

