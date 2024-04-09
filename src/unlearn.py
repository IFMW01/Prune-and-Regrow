import torch.optim as optim
import torch
import torch.nn as nn
import json
import os
import unlearning_methods as um
import training as tr
import load_datasets as ld
import membership_inference as mi
import glob
import utils 

def main(config):
    dataset_pointer = config.get("dataset_pointer", None)
    pipeline = config.get("pipeline", None)
    architecture = config.get("architecture", None)
    n_epochs = config.get("n_epochs", None)
    seeds = config.get("seeds", None)
    n_classes = config.get("n_classes", None)
    n_inputs = config.get("n_inputs", None)
    unlearning = config.get("unlearning", None)
    n_impair = config.get("n_impair", None)
    n_repair = config.get("n_repair", None)
    n_fine_tine = config.get("n_fine_tine", None)
    forget_instances_num = config.get("forget_instances_num", None)
    pruning_ratio = config.get("pruning_ratio", None)
    
    print("Received arguments from config file:")
    print(f"Unlearning: {unlearning}")
    print(f"Dataset pointer: {dataset_pointer}")
    print(f"pipeline: {pipeline}")
    print(f"Architecture: {architecture}")
    print(f"Number of retrain epochs: {n_epochs}")
    print(f"Seeds: {seeds}")
    print(f"Number of impair epochs: {n_impair}")
    print(f"Number of repair epochs: {n_repair}")
    print(f"Number of fine tuning epochs: {n_fine_tine}")

    device = utils.get_device()
    model_dir = f'TRAIN/{dataset_pointer}/{architecture}'
    if not os.IsADirectoryError(model_dir):
        print(f"There are no models with this {architecture} for this {dataset_pointer} in the TRAIN directory. Please train relevant models")
        return
    else:
        train_set,test_set = ld.load_datasets(dataset_pointer,pipeline,True)
        forget_set, remain_set = um.create_forget_remain_set(forget_instances_num,train_set)
        print("Creating remain and forget data loaders")
        forget_loader, remain_loader, test_loader= ld.loaders(forget_set,remain_set,test_set,ld.collate_fn_SC)
        save_dir = f"TRAIN/{dataset_pointer}/{architecture}/'UNLEARN"
        utils.create_dir(save_dir)
        for seed in seeds:
            model_dir = f'TRAIN/{dataset_pointer}/{architecture}/{seed}'
            save_dir = f"TRAIN/{dataset_pointer}/{architecture}/'UNLEARN/{seed}"
            utils.create_dir(save_dir)
            print(f"Acessing trained model on seed: {seed}")
            model_path = glob.glob(os.path.join(model_dir, '*.pth'))
            naive_model = um.naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,forget_loader,test_loader, n_epochs,seed)
            df_softmax_naive = utils.logits(best_model, train_loader, test_loader,device)
            df_softmax_outputs.to_csv(f'{save_path}softmax_outputs.csv',index = False)

            

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
