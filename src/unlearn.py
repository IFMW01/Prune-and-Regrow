import torch.optim as optim
import torch
import torch.nn as nn
import json
import os
import training as tr
import load_datasets as ld
import membership_inference as mi
import utils 

def main(config):
    dataset_pointer = config.get("dataset_pointer", None)
    architecture = config.get("architecture", None)
    n_epochs = config.get("n_epochs", None)
    seeds = config.get("seeds", None)
    n_classes = config.get("n_classes", None)
    n_inputs = config.get("n_inputs", None)
    unlearning = config.get("unlearning", None)
    n_impair = config.get("n_impair", None)
    n_repair = config.get("n_repair", None)
    n_fine_tine = config.get("n_fine_tine", None)
    
    print("Received arguments from config file:")
    print(f"Unlearning: {unlearning}")
    print(f"Dataset pointer: {dataset_pointer}")
    print(f"Architecture: {architecture}")
    print(f"Number of retrain epochs: {n_epochs}")

    print(f"Seeds: {seeds}")

    device = utils.get_device()
            
    if training == 'Base':
        save_dir = f"{training}_{dataset_pointer}"
        utils.create_dir(save_dir)
        save_dir = os.path.join(save_dir, f"{architecture}")
        utils.create_dir(save_dir)
        train_loader,valid_loader,test_loader = ld.load_datasets(dataset_pointer,pipeline)
        for i in range(len(seeds)):
            save_dir = os.path.join(f"{training}_{dataset_pointer}", f"{architecture}")
            seed = seeds[i]
            utils.set_seed(seed)
            model,optimizer, scheduler,criterion = utils.initialise_model(architecture,n_inputs,n_classes,device)
            save_dir = os.path.join(save_dir, f"{seed}")
            utils.create_dir(save_dir)
            print(save_dir)
            save_path = save_dir + '/'
            create_base_model(model,optimizer,criterion,save_path,device, n_epochs, seed,train_loader,valid_loader,test_loader)
    print("FIN")

if __name__ == "__main__":
    with open("./config.json", "r") as f:
        config = json.load(f)
    main(config)
