import torch
import pandas as pd
import os
import utils
import json
import glob
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def attack_models(num_models,x_train,y_train,x_test,y_test,attack_model,save_dir,device):
  x_train = x_train.to(device)
  y_train = y_train.to(device)
  x_test = x_test.to(device)
  y_test = y_test.to(device)
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

      model = TabNetClassifier(  n_d = 32,
      n_a = 32,seed =i,verbose=0 )
      model.to(device)
      model.fit(x_train, y_train,
      eval_set=[(x_train, y_train),(x_test, y_test)],
      max_epochs = 100,
      patience =100
      )
      save_name = f'tabnet_model_{i}.json'
      save_path = f"{save_dir}/{save_name}"
      torch.save(model, save_path)
       
    print(f"ATTACK MODEL: {save_name} STATS")
    modelstats(model,x_train,x_test,y_train,y_test)


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
  
  label_1 = df[df['label'] == 1].index

  label_0 = df[df['label'] == 0].sample(n=len(label_1), random_state=42).index

  balanced_indices = label_1.union(label_0)
  balanced_df = df.loc[balanced_indices]
  balanced_df.to_csv(f'{data_directory}_all_balanced.csv',index = False)
  df_lables = balanced_df['label']
  balanced_df = balanced_df.drop(['label'], axis=1)

  x_train,x_test,y_train,y_test= train_test_split(balanced_df,df_lables, test_size=0.2, random_state=42,shuffle=True)
  print(x_train)
  print(y_train)
  return x_train,y_train,x_test,y_test

def main(config_attack,config_base):
    dataset_pointer = config_base.get("dataset_pointer", None)
    architecture = config_base.get("architecture", None)
    n_attack_models = config_attack.get("n_attack_models", None)

    device = utils.get_device()
    dataset_dir = f'TRAIN/{dataset_pointer}/{architecture}/MIA'
    if not os.path.exists(dataset_dir):
        print(f"There are no models with this {architecture} for this {dataset_pointer} in the MIA directory. Please train relevant models")
        return
    logit_dir = dataset_dir + '/Logits'
    softmax_dir = dataset_dir + '/Softmax'
    logit_attack = logit_dir +"/Attack"
    softmax_attack = softmax_dir +"/Attack"
    utils.create_dir(logit_attack)
    utils.create_dir(softmax_attack)

    print(logit_dir)

    x_train_logits,y_train_logits,x_test_logits,y_test_logits = create_mia_datasets(logit_dir)
    x_train_loss,y_train_loss,x_test_loss,y_test_loss = create_mia_datasets(softmax_dir)

    print("Logit Attack Models")
    attack_model = 'xgb'
    attack_models(n_attack_models,x_train_logits,y_train_logits,x_test_logits,y_test_logits,attack_model,logit_attack,device)
    attack_model = 'tabnet'
    attack_models(n_attack_models,x_train_logits,y_train_logits,x_test_logits,y_test_logits,attack_model,logit_attack,device)
    print("Loss Attack Models")
    attack_model = 'xgb'
    attack_models(n_attack_models,x_train_loss,y_train_loss,x_test_loss,y_test_loss,attack_model,softmax_attack,device)
    attack_model = 'tabnet'   
    attack_models(n_attack_models,x_train_loss,y_train_loss,x_test_loss,y_test_loss,attack_model,softmax_attack,device)
    print("FIN")

if __name__ == "__main__":
    with open("./configs/base_config.json","r") as b:
        config_base = json.load(b)    
    with open("./configs/attack_config.json", "r") as a:
        config_attack = json.load(a)
    main(config_attack,config_base)
