import torch
import pandas as pd
import os
import utils
import json
import glob
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def attack_models(num_models,x_train,y_train,x_test,y_test,attack_model,device):
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
    elif attack_model == 'tabnet':
       y_train = LabelEncoder.fit_transform(y_train)
       y_test = LabelEncoder.transform(y_test)
       x_train = x_train
       x_test = x_test

       model = TabNetClassifier(  n_d = 32,
       n_a = 32)
       model.fit(x_train, y_train,
        eval_set=[(x_train, y_train),(x_test, y_test)],
        max_epochs = 100,
        patience =100
        )
       
    print(f"ATTACK MODEL: {i} STATS")
    modelstats(model,x_train,x_test,y_train,y_test)
    save_path = f"xgboost_model_{i}.json"
    model.save_model(save_path)

def modelstats(model,x_train,x_test,y_train,y_test):
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test)

  train_accuracy = accuracy_score(y_train, y_pred_train)
  test_accuracy = accuracy_score(y_test, y_pred_test)

  train_precision = precision_score(y_train, y_pred_train)
  test_precision = precision_score(y_test, y_pred_test)

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
    list_of_files = glob.glob(f'{data_directory}*.csv')
    for  data_path in list_of_files:
        df = pd.concat([df,pd.read_csv(data_path)],ignore_index=True)
    df.to_csv(f'{data_directory}_all.csv',index = False)

    df_lables = df['label']
    df.drop(['label'], axis=1)

    x_train,y_train,x_test,y_test= train_test_split(df,df_lables, test_size=0.2, random_state=42,shuffle=True)

    return x_train,y_train,x_test,y_test

def main(config):
    dataset_pointer = config.get("dataset_pointer", None)
    architecture = config.get("architecture", None)
    n_attack_models = config.get("n_attack_models", None)

    device = utils.get_device()
    dataset_dir = f'TRAIN/{dataset_pointer}/{architecture}/MIA'
    if not os.path.exists(dataset_dir):
        print(f"There are no models with this {architecture} for this {dataset_pointer} in the MIA directory. Please train relevant models")
        return
    logit_dir = dataset_dir + '/Logits'
    softmax_dir = dataset_dir + '/Softmax'

    x_train_logits,y_train_logits,x_test_logits,y_test_logits = create_mia_datasets(logit_dir)
    x_train_loss,y_train_loss,x_test_loss,y_test_loss = create_mia_datasets(softmax_dir)

    print("Logit Attack Models")
    attack_model = 'xgb'
    attack_models(n_attack_models,x_train_logits,y_train_logits,x_test_logits,y_test_logits,attack_model,device)
    attack_model = 'tabnet'
    attack_models(n_attack_models,x_train_logits,y_train_logits,x_test_logits,y_test_logits,attack_model,device)
    print("Softmax Attack Models")
    attack_model = 'xgb'
    attack_models(n_attack_models,x_train_logits,y_train_logits,x_test_logits,y_test_logits,attack_model,device)
    attack_model = 'tabnet'   
    attack_models(n_attack_models,x_train_logits,y_train_logits,x_test_logits,y_test_logits,attack_model,device)
    print("FIN")

if __name__ == "__main__":
    with open("./configs/attack_config.json", "r") as f:
        config = json.load(f)
    main(config)
