import torch
import pandas as pd
import os
import utils
import json
import glob
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import models.attack_model as attack_model
import utils
from torch.utils.data import DataLoader
from Trainer import Trainer
import random

def attack_models_old(num_models,x_train,y_train,x_test,y_test,attack_model,save_dir,device):
  for i in range(num_models):
    utils.set_seed(i)
    if attack_model == 'xgb':
        params = {
            'learning_rate': 0.05,
            'n_estimators': 150,
            'max_depth': 8,
            'min_child_weight': 0.5,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 1,
            'random_state': i,
            'device':device
        }
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        save_name = f'xgboost_model_{i}.model'
        save_path = f"{save_dir}/{save_name}"
        model.save_model(save_path)
    elif attack_model == 'tabnet':
      if not isinstance(x_train, np.ndarray):
        x_train = x_train.values
        x_test = x_test.values
        y_train = y_train.values
        y_test = y_test.values
        
      # x_train = x_train.to(device)
      # y_train = y_train.to(device)
      # x_test = x_test.to(device)
      # y_test = y_test.to(device)

      model = TabNetClassifier(  n_d = 32,
      n_a = 32,seed =i,verbose=1 )
      model
      model.fit(x_train, y_train,
      eval_set=[(x_train, y_train),(x_test, y_test)],
      max_epochs = 50,
      patience =50,
      eval_metric= 'auc'
      )
      save_name = f'tabnet_model_{i}.pth'
      save_path = f"{save_dir}/{save_name}"
      torch.save(model, save_path)
       
    print(f"ATTACK MODEL: {save_name} STATS")
    modelstats(model,x_train,x_test,y_train,y_test)

def create_attack_model(num_models,train_loader,test_loader,n_inputs,save_dir,device,dict):
  criterion = nn.CrossEntropyLoss()    
  for i in range(num_models):
    dict[f'{i}'] = {}
    utils.set_seed(i)
    model = attack_model.softmax_net(n_inputs)
    optimizer = optim.Adam(model.parameters(),0.001)
    trainer = Trainer(model, train_loader, train_loader, test_loader, optimizer, criterion, device, 50,2,i)
    best_model,best_train_accuracy,best_train_loss,best_train_ece,best_test_accuracy,best_test_loss,best_test_ece,best_model_epoch,best_time = trainer.train()
    dict = utils.update_dict(dict[f'{i}'],best_time,best_model_epoch,best_train_accuracy,best_train_loss,best_train_ece,best_test_accuracy,best_test_loss,best_test_ece)
    save_name = f'attack_model_{i}.pth'
    save_path = f"{save_dir}/{save_name}"
    torch.save(best_model, save_path)
  return dict


def modelstats(model,x_train,x_test,y_train,y_test):
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test)

  train_accuracy = accuracy_score(y_train, y_pred_train)
  test_accuracy = accuracy_score(y_test, y_pred_test)

  train_precision = precision_score(y_train, y_pred_train,zero_division=1)
  test_precision = precision_score(y_test, y_pred_test,zero_division=1)

  train_recall = recall_score(y_train, y_pred_train)
  test_recall = recall_score(y_test, y_pred_test)

  train_f1 = f1_score(y_train, y_pred_train)
  test_f1 = f1_score(y_test, y_pred_test)

  print("Training Accuracy:", train_accuracy)
  print("Testing Accuracy:", test_accuracy)
  print("Training Precision:", train_precision)
  print("Testing Precision:", test_precision)
  print("Training Recall:", train_recall)
  print("Testing Recall:", test_recall)
  print("Training F1 Score:", train_f1)
  print("Testing F1 Score:", test_f1)


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
  dataset_dir = f'TRAIN/{dataset_pointer}/{architecture}/MIA'
  if not os.path.exists(dataset_dir):
      print(f"There are no models with this {architecture} for this {dataset_pointer} in the MIA directory. Please train relevant models")
      return
  logit_dir = dataset_dir + '/Logits'
  loss_dir = dataset_dir + '/Loss'
  logit_attack = logit_dir +"/Attack"
  loss_attack = loss_dir +"/Attack"
  utils.create_dir(logit_attack)
  utils.create_dir(loss_attack)
  results_dict = {}
  results_dict['Logits'] = {}

  train_set_logits,test_set_logits = create_mia_datasets(logit_dir)
  train_set_loss,test_set_loss = create_mia_datasets(loss_dir)
  train_logits,test_logits = create_mia_loader(train_set_logits,test_set_logits)
  train_loss,test_loss = create_mia_loader(train_set_loss,test_set_loss)
  results_dict['Logits'] = {}
  results_dict['Loss'] = {}
  print("Logit Attack Models")
  results_dict['Logits'] = create_attack_model(n_attack_models,train_logits,test_logits,n_classes,logit_attack,device,results_dict['Logits'])
  print("Loss Attack Models")
  results_dict['Loss'] = create_attack_model(n_attack_models,train_loss,test_loss,1,loss_attack,device,results_dict['Loss'])
  with open(f"{dataset_dir}/attack_model_results.json",'w') as f:
    json.dump(results_dict,f)
  print("FIN")

if __name__ == "__main__":
  with open("./configs/base_config.json","r") as b:
      config_base = json.load(b)    
  with open("./configs/attack_config.json", "r") as a:
      config_attack = json.load(a)
  main(config_attack,config_base)
