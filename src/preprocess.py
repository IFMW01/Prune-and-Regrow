import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

def pad_datum(max,datum):
    pad_datum =nn.ConstantPad1d((0, max - datum[0].shape[1]), 0)(datum[0])
    return pad_datum

def convert_waveform(dataset,pipeline,resample:bool,orig_freq=16000,new_freq=8000):
  converted_datset = []
  if resample == True:
     resample_transform = torchaudio.transforms.Resample(orig_freq, new_freq)
     max = new_freq
  else:
     max = orig_freq
  for datum in tqdm(dataset):
    datum_0 = datum[0]
    if datum[0].shape[1]<max:
        datum_0 = pad_datum(max,datum)
    if pipeline:
        if resample:
            converted_datset.append(pipeline(resample_transform(datum_0)),datum[1], datum[2],datum[3], datum[4])
        else:   
            converted_datset.append(((pipeline(datum_0)),datum[1], datum[2],datum[3], datum[4]))
    else:
      converted_datset.append(((datum_0),datum[1], datum[2],datum[3], datum[4]))
  return converted_datset