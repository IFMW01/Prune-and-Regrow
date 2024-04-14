import torchaudio
import os
import preprocess as pp
import torch
import librosa
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS

labels = np.load('./labels/lables.npy')
labels = labels.tolist()

def load_datasets(dataset_pointer :str,pipeline:str,unlearnng:bool):

    if pipeline == 'mel':
        pipeline_on_wav = WavToMel()
    elif pipeline =='spec':
        pipeline_on_wav = WavToSpec()
    if not os.path.exists(dataset_pointer):
            print(f"Downloading: {dataset_pointer}")
    if dataset_pointer == 'SpeechCommands':
        train_list = SubsetSC("testing") 
        test_list = SubsetSC("testing")
    else:
        return
    
    train_set,test_set = convert_sets(train_list,test_list,pipeline_on_wav)

    if unlearnng:
            return train_set,test_set
    else:
        train_loader = loaders(train_set,dataset_pointer)
        train_eval_loader = loaders(train_set,dataset_pointer,16384)
        test_loader = loaders(test_set,dataset_pointer,16384)
        return train_loader,train_eval_loader,test_loader

    

def convert_sets(train_list,test_list,pipeline_on_wav):

    print("Converting datasets")
    train_set = pp.convert_waveform(train_list,pipeline_on_wav,False)
    test_set = pp.convert_waveform(test_list,pipeline_on_wav,False)

    return train_set,test_set

def load_mia_dataset(dataset_pointer :str,pipeline:str):
    
    if pipeline == 'mel':
        pipeline_on_wav = WavToMel()
    elif pipeline =='spec':
        pipeline_on_wav = WavToSpec()

    if dataset_pointer == 'SpeechCommands':
        if not os.path.exists(dataset_pointer):
            print(f"Downloading: {dataset_pointer}")
        all_list = SubsetSC("all")
        print("Converting All Set")
        all_set = pp.convert_waveform(all_list,pipeline_on_wav,False)
        return all_set
    else:
        return

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
        n_fft=1024,
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

def loaders(dataset,dataset_pointer,batch_size=256):
  if dataset_pointer == 'SpeechCommands':
      collate_fn = collate_fn_SC
  else:
      return
  
  dataset_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      collate_fn = collate_fn
  )

  return dataset_loader
    
