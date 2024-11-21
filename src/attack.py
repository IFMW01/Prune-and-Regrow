import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import utils
import json
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import models.attack_model as attack_model
import utils
from Trainer import Trainer
import random

# Creates attack models and saves then in predefined folders that are created 

def create_attack_model(num_models,train_loader,test_loader,n_inputs,save_dir,device,dict):
  criterion = nn.CrossEntropyLoss()    
  for i in range(num_models):
    dict[f'{i}'] = {}
    utils.set_seed(i)
    model = attack_model.softmax_net(n_inputs)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),0.001)
    trainer = Trainer(model, train_loader, train_loader, test_loader, optimizer, criterion, device, 50,2,i)
    best_model,best_train_accuracy,best_train_loss,best_train_ece,best_test_accuracy,best_test_loss,best_test_ece,best_model_epoch,best_time = trainer.train()
    dict[f'{i}'] = utils.update_dict(dict[f'{i}'],best_time,best_model_epoch,best_train_accuracy,best_train_loss,best_train_ece,best_test_accuracy,best_test_loss,best_test_ece)
    save_name = f'attack_model_{i}.pth'
    save_path = f"{save_dir}/{save_name}"
    torch.save(best_model, save_path)
  return dict

# Creates balanced datatset to train attack model from loss outputs of trained models
def create_mia_datasets(data_directory):
  df = pd.DataFrame()
  list_of_files = glob.glob(f'{data_directory}/*.csv')
  for  data_path in list_of_files:
    df = pd.concat([df,pd.read_csv(data_path)],ignore_index=True)

  label_1 = df[df['label'] == 1].index.tolist()
  label_0 = df[df['label'] == 0].index.tolist()
  random.seed(42)
  label_0 = random.sample(label_0,len(label_1))
  balanced_indices = label_1 +label_0
  balanced_df = df.loc[balanced_indices]
  balanced_df = balanced_df.sample(frac=1, random_state=42)

  balanced_df.to_csv(f'{data_directory}_all_balanced.csv',index = False)
  df_lables = balanced_df['label']
  balanced_df = balanced_df.drop(['label'], axis=1)

  x_train,x_test,y_train,y_test= train_test_split(balanced_df,df_lables, test_size=0.2)
  x_train = torch.tensor(x_train.values,dtype=torch.float)
  x_test = torch.tensor(x_test.values,dtype=torch.float)
  y_train = torch.tensor(y_train.values,dtype=torch.long)
  y_test = torch.tensor(y_test.values,dtype=torch.long)
  train_set = [(x_train[i], y_train[i]) for i in range(len(x_train))]
  test_set = [(x_test[i], y_test[i]) for i in range(len(x_test))]

  return train_set,test_set

def create_mia_loader(train_set,test_set):
  train_loader = DataLoader(train_set, batch_size=264, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=264, shuffle=False)
  return train_loader,test_loader

def main(config_attack,config_base):
  dataset_pointer = config_base.get("dataset_pointer", None)
  architecture = config_base.get("architecture", None)
  n_classes = config_base.get("n_classes", None)
  n_attack_models = config_attack.get("n_attack_models", None)

  device = utils.get_device()
  dataset_dir = f'Results/{dataset_pointer}/{architecture}/MIA'
  if not os.path.exists(dataset_dir):
      print(f"There are no models with this {architecture} for this {dataset_pointer} in the MIA directory. Please train relevant models")
      return
  loss_dir = dataset_dir + '/Attack'
  utils.create_dir(loss_dir)
  results_dict = {}
  train_set_loss,test_set_loss = create_mia_datasets(dataset_dir)
  train_loss,test_loss = create_mia_loader(train_set_loss,test_set_loss)
  results_dict['Loss'] = {}
  print("Loss Attack Models")
  results_dict['Loss'] = create_attack_model(n_attack_models,train_loss,test_loss,1,loss_dir,device,results_dict['Loss'])
  with open(f"{dataset_dir}/attack_model_results.json",'w') as f:
    json.dump(results_dict,f)
  print("FIN")

if __name__ == "__main__":
  with open("./configs/base_config.json","r") as b:
      config_base = json.load(b)    
  with open("./configs/attack_config.json", "r") as a:
      config_attack = json.load(a)
  main(config_attack,config_base)
