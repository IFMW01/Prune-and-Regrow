import torch.optim as optim
import torch
import torch.nn as nn
import json
import os
import training as tr
import load_datasets as ld
import membership_inference as mi
import utils 

from vgg import VGGish,VGG9
# from transformer import SimpleViT

def create_base_model(model,optimizer, scheduler,criterion,dataset_pointer,pipeline,save_path,device, n_epochs, seed):
        train_loader,valid_loader,test_loader = ld.load_datasets(dataset_pointer,pipeline)
        best_model,accuracies = tr.train(model, train_loader,valid_loader, test_loader, optimizer, criterion, device, n_epochs, seed)
        torch.save(best_model, f"{save_path}.pth")
        df_softmax_outputs = mi.mai_logits(best_model, train_loader, test_loader,device)
        df_softmax_outputs.to_csv(f'{save_path}_softmax_outputs.csv',index = False)


def main(config):
    dataset_pointer = config.get("dataset_pointer", None)
    pipeline = config.get("pipeline", None)
    architecture = config.get("architecture", None)
    n_epochs = config.get("n_epochs", None)
    training = config.get("training", None)
    seeds = config.get("seeds", None)
    n_classes = config.get("n_classes", None)
    n_inputs = config.get("n_inputs", None)

    print("Received arguments from config file:")
    print(f"Dataset pointer: {dataset_pointer}")
    print(f"Pipeline: {pipeline}")
    print(f"Architecture: {architecture}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Training: {training}")
    print(f"Seeds: {seeds}")

    device = utils.get_device()
            
    if training == 'Base':
        save_dir = f"{training}_{dataset_pointer}"
        utils.create_dir(save_dir)
        for i in range(len(seeds)):
            seed = seeds[i]
            utils.set_seed(seed)
            model,optimizer, scheduler,criterion = utils.initialise_model(architecture,n_inputs,n_classes,device)
            save_dir = os.path.join(f"{training}_{dataset_pointer}", f"{seed}")
            utils.create_dir(save_dir)
            save_path = f"{save_dir}\{architecture}_{seed}"
            create_base_model(model,optimizer, scheduler,criterion,dataset_pointer,pipeline,save_path,device, n_epochs, seed)
    print("FIN")

if __name__ == "__main__":
    with open("./config.json", "r") as f:
        config = json.load(f)
    main(config)