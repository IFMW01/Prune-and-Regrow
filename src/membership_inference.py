from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import load_datasets as ld
import Trainer
from Trainer import Trainer
import json
import os
import utils
from vgg import VGGish, VGG9
from tqdm import tqdm

def create_membership_inference_dataset(all_processed,seed):
  train_set, test_set = train_test_split(all_processed,train_size = 0.5, test_size=0.1, random_state=seed)
  return train_set,test_set

def membership_inference_attack(dataset_pointer,architecture,n_input,n_classes,pipeline,device,n_shadow_models,n_shadow_epochs,save_dir):
  test_acc = 0 
  test_loss = float('inf')
  for seed in range(n_shadow_models):

    all_processed = ld.load_mia_dataset(dataset_pointer,pipeline)
    train_set_mia,test_set_mia = create_membership_inference_dataset(all_processed,seed)
    train_loader,train_eval_loader,test_loader=  ld.loaders(train_set_mia,test_set_mia,dataset_pointer)
    model,optimizer,criterion = utils.initialise_model(architecture,n_input,n_classes,device)
    trainer = Trainer(model, train_loader, train_eval_loader, test_loader, optimizer, criterion, device, n_shadow_epochs,n_classes,seed)
    mia_model,mia_test_accuracy,mia_test_loss= trainer.train()
    test_acc += mia_test_accuracy
    test_loss += mia_test_loss
    mia_logit_df = utils.logits(mia_model, train_eval_loader, test_loader,device)
    filename = (f"MAI {seed}.csv")
    mia_logit_df.to_csv(f"{save_dir}/{filename}", index = False)
    print(f"{filename} saved")

  print(f"Average attack test accuracy: {(test_acc/n_shadow_models):.4f}")
  print(f"Average attack test loss: {(test_acc/n_shadow_models):.4f}")

def main(config):
    dataset_pointer = config.get("dataset_pointer", None)
    pipeline = config.get("pipeline", None)
    architecture = config.get("architecture", None)
    n_classes = config.get("n_classes", None)
    n_inputs = config.get("n_inputs", None)
    n_shadow_models = config.get("n_shadow_models", None)
    n_shadow_epochs = config.get("n_shadow_epochs", None)
    

    device = utils.get_device()
    save_dir = f'TRAIN/{dataset_pointer}/{architecture}/MIA'
    utils.create_dir(save_dir)
    membership_inference_attack(dataset_pointer,architecture,n_inputs,n_classes,pipeline,device,n_shadow_models,n_shadow_epochs,save_dir)
    print("FIN")

if __name__ == "__main__":
    with open("./configs/mia_config.json", "r") as f:
        config = json.load(f)
    main(config)

