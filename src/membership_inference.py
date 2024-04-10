from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import utils
from tqdm import tqdm

def create_membership_inference_dataset(all_mel,seed):
  train, test = train_test_split(all_mel,train_size = 0.5, test_size=0.1, random_state=seed)
  return train,test

