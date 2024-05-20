import json
import os
import glob
import utils 
import math
import random
import numpy as np
import torch
import os.path
from unlearn import unlearning_methods as um
from unlearn import unlearn_metrics
from unlearn import Unlearner
import unlearn
from torch.utils.data import DataLoader
from datasets_unlearn import load_datasets as ld




def param_regrowth(remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,dataset_pointer,architecture,n_epochs,seeds,n_classes,n_inputs,n_epoch_impair,n_epoch_repair,n_epochs_fine_tune,forget_amount,pruning_ratio,tag,device):
            
    results_dict = {}
    for seed in seeds:
        print(f'\nSeed: {seed}')
        results_dict[seed] = {}
        model_dir = f'Results/{dataset_pointer}/{architecture}/{seed}'
        save_dir = f"Results/{dataset_pointer}/{architecture}/UNLEARN/{tag}/{forget_amount}/{seed}/"
        utils.create_dir(save_dir)
        print(f"Acessing trained model on seed: {seed}")
        model_path = glob.glob(os.path.join(model_dir,'*.pth'))
        model_path = model_path[0]

        results_dict[seed]["OMP Unlearning"] = {}
                                                        
        omp_model,results_dict[seed]["OMP Unlearning"] = um.omp_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,pruning_ratio,n_epoch_repair,results_dict[seed]["OMP Unlearning"],n_classes,architecture,seed)

        results_dict[seed]["Kurtosis Unlearning"] = {} 
        kk_model,results_dict[seed]["Kurtosis Unlearning"] = um.kurtosis_of_kurtoses_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_repair,results_dict[seed]["Kurtosis Unlearning"],n_classes,architecture,seed)

        print(f'All unlearning methods applied for seed: {seed}.\n{results_dict}')

    
def main(config_unlearn,config_base):
    dataset_pointer = config_base.get("dataset_pointer",None)
    pipeline = config_base.get("pipeline",None)
    architecture = config_base.get("architecture",None)
    n_epochs = config_base.get("n_epochs",None)
    seeds = config_base.get("seeds",None)
    n_classes = config_base.get("n_classes",None)
    n_inputs = config_base.get("n_inputs",None)
    unlearning = config_unlearn.get("unlearning",None)
    n_epoch_impair = config_unlearn.get("n_epoch_impair",None)
    n_epoch_repair = config_unlearn.get("n_epoch_repair",None)
    n_epochs_fine_tune = config_unlearn.get("n_epochs_fine_tune",None)
    forget_random = config_unlearn.get("forget_random",None)
    forget_percentage = config_unlearn.get("forget_percentage",None)
    forget_classes = config_unlearn.get("forget_classes",None)
    forget_classes_num = config_unlearn.get("forget_classes_num",None)
    pruning_ratio = config_unlearn.get("pruning_ratio",None)
    
    print("Received arguments from config file:")
    print(f"Unlearning: {unlearning}")
    print(f"Dataset pointer: {dataset_pointer}")
    print(f"pipeline: {pipeline}")
    print(f"Architecture: {architecture}")
    print(f"Number of retrain epochs: {n_epochs}")
    print(f"Seeds: {seeds}")
    print(f"Number of impair epochs: {n_epoch_impair}")
    print(f"Number of repair epochs: {n_epoch_repair}")
    print(f"Number of fine tuning epochs: {n_epochs_fine_tune}")
    print(f"Forgetting random samples : {forget_random}")
    if forget_random == True:
        tag = 'Item_Removal'
        print(f"Percentage of data to forget: {forget_percentage}")
    print(f"Forgetting classes : {forget_classes}")
    if  forget_classes == True:
        tag = 'Class_Removal'
        print(f"Number of classes to forget: {forget_classes_num}")
    print(f"Pruning ratio: {pruning_ratio}")
    device = utils.get_device()

    model_dir = f'Results/{dataset_pointer}/{architecture}'
    if not os.path.exists(model_dir):
        print(f"There are no models with this {architecture} for this {dataset_pointer} in the TRAIN directory. Please train relevant models")
        return
    else:
        if forget_random == True:
            remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,forget_number = unlearn.forget_rand_datasets(dataset_pointer,pipeline,forget_percentage,device,n_classes) 
        elif forget_classes == True:
            remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,forget_number = unlearn.forget_class_datasets(dataset_pointer,pipeline,forget_classes_num,n_classes,device) 
        param_regrowth(remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,dataset_pointer,architecture,n_epochs,seeds,n_classes,n_inputs,n_epoch_impair,n_epoch_repair,n_epochs_fine_tune,forget_number,pruning_ratio,tag,device)
    print("FIN")

if __name__ == "__main__":
    with open("./configs/base_config.json","r") as b:
        config_base = json.load(b)    
    with open("./configs/unlearn_config.json","r") as u:
        config_unlearn = json.load(u)
    main(config_unlearn,config_base)
