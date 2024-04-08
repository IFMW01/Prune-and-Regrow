import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

def pad_datum(datum):
    pad_datum =nn.ConstantPad1d((0, max - datum[0].shape[1]), 0)(datum[0])
    return pad_datum

def convert_waveform(dataset:list,pipeline,resample:bool,orig_freq=16000,new_freq=8000):
  converted_datset = []
  if resample:
     resample = torchaudio.transforms.Resample(orig_freq, new_freq)
     max = new_freq
  else:
     max = orig_freq
  for datum in tqdm(dataset):
    if datum[0].shape[1]<max:
        datum = pad_datum(datum)
    if pipeline:
        if resample:
            converted_datset.append((pipeline(resample(datum[0])),datum[1], datum[2],datum[3], datum[4] ))
        else:   
            converted_datset.append((pipeline(datum)),datum[1], datum[2],datum[3], datum[4] )
    return converted_datset