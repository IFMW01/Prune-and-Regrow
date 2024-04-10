import torch
import json
import os
import training as tr
import load_datasets as ld
import utils 

from vgg import VGGish,VGG9
# from transformer import SimpleViT

def create_base_model(model,optimizer,criterion,save_path,device, n_epochs, seed,train_loader,test_loader):
        best_model,acc,loss = tr.train(model, train_loader, test_loader, optimizer, criterion, device, n_epochs, seed)
        torch.save(best_model, f"{save_path}Model_{acc}_{loss}.pth")
        df_softmax_outputs = utils.logits(best_model, train_loader, test_loader,device)
        df_softmax_outputs.to_csv(f'{save_path}softmax_outputs.csv',index = False)


def main(config):
    dataset_pointer = config.get("dataset_pointer", None)
    pipeline = config.get("pipeline", None)
    architecture = config.get("architecture", None)
    n_epochs = config.get("n_epochs", None)
    seeds = config.get("seeds", None)
    n_classes = config.get("n_classes", None)
    n_inputs = config.get("n_inputs", None)

    print("Received arguments from config file:")
    print(f"Dataset pointer: {dataset_pointer}")
    print(f"Pipeline: {pipeline}")
    print(f"Architecture: {architecture}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Seeds: {seeds}")

    device = utils.get_device()
    save_dir = 'TRAIN'
    utils.create_dir(save_dir)
    save_dir = os.path.join(save_dir, f"{dataset_pointer}")
    utils.create_dir(save_dir)
    save_dir = os.path.join(save_dir, f"{architecture}")
    utils.create_dir(save_dir)
    train_loader,test_loader = ld.load_datasets(dataset_pointer,pipeline,False)
    for seed in seeds:
        save_dir = os.path.join(f"TRAIN/{dataset_pointer}/{architecture}")
        utils.set_seed(seed)
        model,optimizer, scheduler,criterion = utils.initialise_model(architecture,n_inputs,n_classes,device)
        save_dir = os.path.join(save_dir, f"{seed}")
        utils.create_dir(save_dir)
        print(save_dir)
        save_path = save_dir + '/'
        create_base_model(model,optimizer,criterion,save_path,device, n_epochs, seed,train_loader,test_loader)
    print("FIN")

if __name__ == "__main__":
    with open("./config.json", "r") as f:
        config = json.load(f)
    main(config)