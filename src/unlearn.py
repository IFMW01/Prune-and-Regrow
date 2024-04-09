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

def unlearn_logits(model,forget_loader,device,save_dir,filename):
    logits = utils.logits_unlearn(model,forget_loader,device)
    logits.to_csv(f'{save_dir}{filename}.csv',index = False)


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
    if not os.path.exists(model_dir):
        print(f"There are no models with this {architecture} for this {dataset_pointer} in the TRAIN directory. Please train relevant models")
        return
    else:
        train_set,test_set = ld.load_datasets(dataset_pointer,pipeline,True)
        forget_set, remain_set = um.create_forget_remain_set(forget_instances_num,train_set)
        print("Creating remain and forget data loaders")
        forget_loader, remain_loader, test_loader= ld.loaders(forget_set,remain_set,test_set,ld.collate_fn_SC)
        save_dir = f"TRAIN/{dataset_pointer}/{architecture}/'UNLEARN/{forget_instances_num}"
        utils.create_dir(save_dir)
        for seed in seeds:

            model_dir = f'TRAIN/{dataset_pointer}/{architecture}/{seed}'
            save_dir = f"TRAIN/{dataset_pointer}/{architecture}/UNLEARN/{forget_instances_num}/{seed}/"
            utils.create_dir(save_dir)
            print(f"Acessing trained model on seed: {seed}")
            model_path = glob.glob(os.path.join(model_dir, '*.pth'))
            model_path = model_path[0]
            orginal_model,optimizer,criterion = um.load_model(model_path,architecture,n_inputs,n_classes,device)
            unlearn_logits(orginal_model,forget_loader,device,save_dir,'orginal_model')



            naive_model = um.naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,forget_loader,test_loader, n_epochs,seed)
            unlearn_logits(naive_model,forget_loader,device,save_dir,'naive_model')
        print("FIN")

if __name__ == "__main__":
    with open("./config.json", "r") as f:
        config = json.load(f)
    main(config)
