from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

def create_membership_inference_dataset(all_mel,seed):
  train, test_valid = train_test_split(all_mel,train_size = 0.5, test_size=0.5, random_state=seed)
  valid, test = train_test_split(test_valid,train_size = 0.5, test_size=0.5, random_state=seed)
  return train,valid,test

def mai_logits(model, train_loader, test_loader,device):
    model.to(device)

    model.eval()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_all = pd.DataFrame()

    # Process training set
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            logits_train = model(data)
            logits_softmax = nn.Softmax(dim=1)(logits_train)
            numpy_logits_train = logits_softmax.cpu().numpy()
            df_logit_train = pd.DataFrame(numpy_logits_train)
            df_train = pd.concat([df_train, df_logit_train], ignore_index=True)
    df_train['label'] = 0

    # Process test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            logits_test = model(data)
            logit_test_softmax = nn.Softmax(dim=1)(logits_test)
            numpy_logits_test = logit_test_softmax.cpu().numpy()
            df_logit_test = pd.DataFrame(numpy_logits_test)
            df_test = pd.concat([df_test, df_logit_test], ignore_index=True)
    df_test['label'] = 1

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    return df_all