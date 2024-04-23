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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchaudio.datasets import SPEECHCOMMANDS

seed = 42

def load_datasets(dataset_pointer :str,pipeline:str,unlearnng:bool):
    global labels
    if pipeline == 'mel':
        pipeline_on_wav = WavToMel()
    elif pipeline =='spec':
        pipeline_on_wav = WavToSpec()
    if not os.path.exists(dataset_pointer):
            print(f"Downloading: {dataset_pointer}")
    if dataset_pointer == 'SpeechCommands':
        train_list = SubsetSC("testing") 
        test_list = SubsetSC("testing")
        labels = np.load('./labels/speech_commands_labels.npy')
        labels = labels.tolist()
        train_set,test_set = convert_sets(train_list,test_list,dataset_pointer,pipeline_on_wav)
    elif dataset_pointer == 'audioMNIST':
        all_list = audioMNIST.create_audioMNIST(pipeline,pipeline_on_wav,dataset_pointer)
        train_set, test_set = audioMNIST.train_test(all_list,pipeline,dataset_pointer,seed)
        labels = np.load('./labels/audiomnist_labels.npy')
        labels = labels.tolist()
    elif dataset_pointer == 'Ravdess':
        all_list = ravdess.create_ravdess(pipeline,pipeline_on_wav,dataset_pointer)
        train_set, test_set = ravdess.train_test(all_list,pipeline,dataset_pointer,seed)
        labels = np.load('./labels/ravdess_label.npy')
    elif dataset_pointer == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    if unlearnng:
        return train_set,test_set
    
    if dataset_pointer == 'audioMNIST' or dataset_pointer == 'Ravdess':
        train_set = DatasetProcessor(train_set)
        test_set = DatasetProcessor(test_set)

    if dataset_pointer == 'SpeechCommands':
        train_loader = trainset_loader(train_set,dataset_pointer)
        train_eval_loader = testset_loader(train_set,dataset_pointer)
        test_loader = testset_loader(test_set,dataset_pointer)

    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                                shuffle=True, num_workers=2)
        train_eval_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                                shuffle=False, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                                shuffle=False, num_workers=2)
        
    return train_loader,train_eval_loader,test_loader


def convert_sets(train_list,test_list,dataset_pointer,pipeline_on_wav):

    print("Converting datasets")
    train_set = pp.convert_waveform(train_list,dataset_pointer,pipeline_on_wav,False)
    test_set = pp.convert_waveform(test_list,dataset_pointer,pipeline_on_wav,False)

    return train_set,test_set

def load_mia_dataset(dataset_pointer :str,pipeline_on_wav):

    if dataset_pointer == 'SpeechCommands':
        if not os.path.exists(dataset_pointer):
            print(f"Downloading: {dataset_pointer}")
        all_list = SubsetSC("all")
    elif dataset_pointer == 'audioMNIST':
        all_list = AudioMNIST.audioMNIST_all()

    print("Converting All Set")
    all_set = pp.convert_waveform(all_list,pipeline_on_wav,False)
    return all_set

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
        elif subset == "all":
            self._walker = [w for w in self._walker]

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
        n_fft=1024,
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

def label_to_index(word):
    return torch.tensor(labels.index(word))

def index_to_label(index):
    return labels[index]

def collate_fn_SC(batch):
    tensors, targets = [], []

    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
    targets = torch.stack(targets)
    tensors = torch.stack(tensors)
    return tensors, targets

def collate_fn_MNIST(batch):
    tensors, targets = [], []

    for waveform,_,_, label, in batch:
        waveform = waveform[None,:,:]
        tensors += [waveform]
        targets += [torch.tensor(label)]
    targets = torch.stack(targets)
    tensors = torch.stack(tensors)
    return tensors, targets

def trainset_loader(dataset,dataset_pointer,batch_size=256):
  if dataset_pointer == 'SpeechCommands':
      collate_fn = collate_fn_SC
  elif dataset_pointer =='audioMNIST':
      collate_fn = collate_fn_MNIST

  dataset_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      collate_fn = collate_fn
  )

  return dataset_loader

def testset_loader(dataset,dataset_pointer,batch_size=256):
  if dataset_pointer == 'SpeechCommands':
      collate_fn = collate_fn_SC
  elif dataset_pointer =='audioMNIST':
      collate_fn = collate_fn_MNIST
  
  dataset_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2,
      pin_memory=True,
      collate_fn = collate_fn
  )

  return dataset_loader
    
