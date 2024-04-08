import torchaudio
import os
import preprocess as pp
import torch
import librosa
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS

def load_datasets(dataset_pointer :str,pipeline:str,batch_size=256):
    if pipeline == 'mel':
        pipeline_on_wav = WavToMel()
    elif pipeline =='spec':
        pipeline_on_wav = WavToSpec()

    if dataset_pointer == 'SpeechCommands':
        labels = np.load('./labels/lables.npy')
        labels = labels.tolist()
        speech_commands = SubsetSC()
        train_list = speech_commands.get_subset("training")
        test_list = speech_commands.get_subset("testing")
        valid_list = speech_commands.get_subset("validation")
        print(train_list[0])
        print("Converting Train Set")
        train_set = pp.convert_waveform(train_list,pipeline_on_wav,False)
        print("Converting Test Set")
        test_set = pp.convert_waveform(test_list,pipeline_on_wav,False)
        print("Converting Validation Set")
        valid_set = pp.convert_waveform(valid_list,pipeline_on_wav,False)

        train_set,valid_set,test_set = loaders(batch_size,train_set,valid_set,test_set,SubsetSC.collate_fn)
        return train_set,valid_set,test_set,labels

    else:
        return

def load_mia_dataset(dataset_pointer :str,pipeline:str,batch_size=256):
    if pipeline == 'mel':
        pipeline_on_wav = WavToMel()
    elif pipeline =='spec':
        pipeline_on_wav = WavToSpec()

    if dataset_pointer == 'SpeechCommands':
        labels = np.load('./SpeechCommands/lables.npy')
        labels = labels.tolist()
        speech_commands = SubsetSC()
        all_list = speech_commands.get_subset("all")
        print("Converting All Set")
        all_set = pp.convert_waveform(all_list,pipeline_on_wav,False)
        return all_set,labels
    else:
        return

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self,labels,subset: str = None):
        super().__init__("./", download=True)
        self.labels = labels

    def load_list(self,filename):
        filepath = os.path.join(self._path, filename)
        with open(filepath) as fileobj:
            return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
    def get_subset(self,subset):
        if subset == "validation":
            self._walker = self.load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = self.load_list("testing_list.txt")
        elif subset == "training":
            excludes = self.load_list("validation_list.txt") + self.load_list("testing_list.txt")
            excludes = set(excludes)
            filepath = os.path.join(self._path, 'training_list.txt')
            self._walker = [w for w in self._walker if w not in excludes]
        elif subset == "all":
            self._walker = [w for w in self._walker]
            for x in range(len(self._walker)):
                self._walker[x] = self._walker[x].replace('SpeechCommands/speech_commands_v0.02/','')
            filepath = os.path.join(self._path, 'all_list.txt')
            with open(filepath, 'w') as f:
                for line in self._walker:
                    f.write(f"{line}\n")
            self._walker = self.__add__load_list("all_list.txt")
    
    def label_to_index(self,word):
        return torch.tensor(self.labels.index(word))

    def index_to_label(labels,index):
        return labels[index]

    def collate_fn(self,batch):
        tensors, targets = [], []

        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]
        targets = torch.stack(targets)
        tensors = torch.stack(tensors)
        return tensors, targets

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

def loaders(batch_size,train_set,valid_set,test_set,collate_fn):
  train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      collate_fn = collate_fn
  )

  valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2,
      pin_memory=True,
      collate_fn = collate_fn
  )

  test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=batch_size,
      shuffle=False,
      drop_last=False,
      num_workers=2,
      pin_memory=True,
      collate_fn = collate_fn
  )
  return train_loader,valid_loader,test_loader
