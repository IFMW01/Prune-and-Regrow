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
from torch.utils.data import DataLoader
from datasets_unlearn import load_datasets as ld

# gets the loss for an unlearned model
def unlearn_logits(model,loader,device,save_dir,filename_loss):
    loss = utils.logits_unlearn(model,loader,device)
    loss.to_csv(f'{save_dir}{filename_loss}.csv',index = False)

# gets the loss for an unlearned model on the ramin, forget and test set
def logit_distributions(model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,filename_loss):
   unlearn_logits(model,remain_eval_loader,device,save_dir,f'{filename_loss}_remain')
   unlearn_logits(model,forget_eval_loader,device,save_dir,f'{filename_loss}_forget')
   unlearn_logits(model,test_loader,device,save_dir,f'{filename_loss}_test')
     
# Creates the data loaders for remain, forget and test set as well as creating the random labelled datalaoder for AM
def create_loaders(remain_set,forget_set,test_set,forget_randl_data):
    remain_loader = DataLoader(remain_set, batch_size=256,
                                        shuffle=True)
    remain_eval_loader = DataLoader(remain_set, batch_size=256,
                                        shuffle=False)
    forget_loader = DataLoader(forget_set, batch_size=256,
                                        shuffle=True)
    forget_eval_loader = DataLoader(forget_set, batch_size=256,
                                        shuffle=False)       
    test_loader = DataLoader(test_set, batch_size=256,
                                        shuffle=False)
    forget_randl_loader = DataLoader(forget_randl_data, batch_size=256,
                                        shuffle=True)
    return remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader


# Calls all unlearning methods to be perfromed on the base models that have been created and saves the results
def unlearning_process(remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,dataset_pointer,architecture,n_epochs,seeds,n_classes,n_inputs,n_epoch_impair,n_epoch_repair,n_epochs_fine_tune,forget_amount,pruning_ratio,tag,device):
            
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
                
        orginal_model,optimizer,criterion = um.load_model(model_path,architecture,0.01,device)
        results_dict[seed]["Original Model"] = {}
        logit_distributions(orginal_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'orginal_model_loss')

        results_dict[seed]["Naive Unlearning"] = {}
        if not os.path.isfile(f"{save_dir}Naive.pth"):
            naive_model,results_dict[seed]["Naive Unlearning"] = um.naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epochs,results_dict[seed]["Naive Unlearning"],seed)
            logit_distributions(naive_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'naive_model_loss')
            torch.save(naive_model,f"{save_dir}Naive.pth")
            results_dict[seed]["Original Model"]["Activation distance"] = unlearn_metrics.actviation_distance(orginal_model, naive_model, forget_eval_loader, device)
            results_dict[seed]["Original Model"]["JS divergance"]  = unlearn_metrics.JS_divergence(orginal_model,naive_model,forget_eval_loader,device)
            loss_results = unlearn_metrics.mia_efficacy(orginal_model,forget_loader,n_classes,device)
            results_dict[seed]["Original Model"]["Loss MIA"] = loss_results

            results_dict[seed]["Naive Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(naive_model, naive_model, forget_eval_loader, device)
            results_dict[seed]["Naive Unlearning"]["JS divergance"] = (unlearn_metrics.JS_divergence(naive_model,naive_model,forget_eval_loader,device)) 
            loss_results = unlearn_metrics.mia_efficacy(naive_model,forget_loader,n_classes,device)
            results_dict[seed]["Naive Unlearning"]["Loss MIA"] =   loss_results 
            with open(f"{save_dir}Niave.json",'w') as f:
                json.dump(results_dict,f)
        else:
            naive_model,optimizer,criterion = um.load_model(f"{save_dir}Naive.pth",architecture,0.01,device)
        
        results_dict[seed]["Gradient Ascent Unlearning"] = {}
        gradient_ascent_model,results_dict[seed]["Gradient Ascent Unlearning"] = um.gradient_ascent(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,device,n_epoch_impair,n_epoch_repair,results_dict[seed]["Gradient Ascent Unlearning"],n_classes,forget_amount,dataset_pointer,architecture,seed)
        logit_distributions(gradient_ascent_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'gradient_ascent_model_loss')
        results_dict[seed]["Gradient Ascent Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(gradient_ascent_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Gradient Ascent Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(gradient_ascent_model,naive_model,forget_eval_loader,device)

        loss_results = unlearn_metrics.mia_efficacy(gradient_ascent_model,forget_loader,n_classes,device)
        results_dict[seed]["Gradient Ascent Unlearning"]["Loss MIA"] =   loss_results    

        results_dict[seed]["Fine Tune Unlearning"] = {}
        fine_tuning_model,results_dict[seed]["Fine Tune Unlearning"] = um.fine_tuning_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epochs_fine_tune,results_dict[seed]["Fine Tune Unlearning"],n_classes,architecture,seed)
        logit_distributions(fine_tuning_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'fine_tuning_model_loss')

        results_dict[seed]["Fine Tune Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(fine_tuning_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Fine Tune Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(fine_tuning_model,naive_model,forget_eval_loader,device)

        loss_results = unlearn_metrics.mia_efficacy(fine_tuning_model,forget_loader,n_classes,device)
        results_dict[seed]["Fine Tune Unlearning"]["Loss MIA"] =   loss_results    

        results_dict[seed]["Stochastic Teacher Unlearning"] = {}
        stochastic_teacher_model,results_dict[seed]["Stochastic Teacher Unlearning"]= um.stochastic_teacher_unlearning(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,device,n_inputs,n_classes,architecture,results_dict[seed]["Stochastic Teacher Unlearning"],n_epoch_impair,n_epoch_repair,seed)
        logit_distributions(stochastic_teacher_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'stochastic_teacher_model_loss')
        results_dict[seed]["Stochastic Teacher Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(stochastic_teacher_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Stochastic Teacher Unlearning"]["JS divergance"]= unlearn_metrics.JS_divergence(stochastic_teacher_model,naive_model,forget_eval_loader,device)     

        loss_results = unlearn_metrics.mia_efficacy(stochastic_teacher_model,forget_loader,n_classes,device)
        results_dict[seed]["Stochastic Teacher Unlearning"]["Loss MIA"] =   loss_results   

        results_dict[seed]["OMP Unlearning"] = {}
                                                        
        omp_model,results_dict[seed]["OMP Unlearning"] = um.omp_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,pruning_ratio,n_epoch_repair,results_dict[seed]["OMP Unlearning"],n_classes,architecture,seed)
        logit_distributions(omp_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'omp_model_loss')

        results_dict[seed]["OMP Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(omp_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["OMP Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(omp_model,naive_model,forget_eval_loader,device)     

        loss_results = unlearn_metrics.mia_efficacy(omp_model,forget_loader,n_classes,device)   
        results_dict[seed]["OMP Unlearning"]["Loss MIA"] =   loss_results    

        results_dict[seed]["Cosine Unlearning"] = {} 
        cosine_model,results_dict[seed]["Cosine Unlearning"] = um.cosine_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_repair,results_dict[seed]["Cosine Unlearning"],n_classes,architecture,seed)
        logit_distributions(cosine_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'cosine_model_loss')
        
        results_dict[seed]["Cosine Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(cosine_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Cosine Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(cosine_model,naive_model,forget_eval_loader,device)  
        loss_results = unlearn_metrics.mia_efficacy(cosine_model,forget_loader,n_classes,device)    
        results_dict[seed]["Cosine Unlearning"]["Loss MIA"] =   loss_results    

        results_dict[seed]["POP"] = {} 
        pop_model,results_dict[seed]["POP"] = um.pop_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_repair,results_dict[seed]["POP"],n_classes,architecture,seed)
        logit_distributions(pop_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'pop_model_loss')

        results_dict[seed]["POP"]["Activation distance"]  = unlearn_metrics.actviation_distance(pop_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["POP"]["JS divergance"]  = unlearn_metrics.JS_divergence(pop_model,naive_model,forget_eval_loader,device)
        loss_results = unlearn_metrics.mia_efficacy(pop_model,forget_loader,n_classes,device) 
        results_dict[seed]["POP"]["Loss MIA"] =   loss_results   

        results_dict[seed]["Amnesiac Unlearning"] = {} 
        randl_model,results_dict[seed]["Amnesiac Unlearning"] = um.amnesiac_unlearning(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,forget_randl_loader,device,n_epoch_impair,n_epoch_repair,results_dict[seed]["Amnesiac Unlearning"],n_classes,architecture,seed)
        logit_distributions(randl_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'randl_model_loss')

        results_dict[seed]["Amnesiac Unlearning"]["Activation distance"]  = unlearn_metrics.actviation_distance(randl_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Amnesiac Unlearning"]["JS divergance"]  = unlearn_metrics.JS_divergence(randl_model,naive_model,forget_eval_loader,device)
        loss_results = unlearn_metrics.mia_efficacy(randl_model,forget_loader,n_classes,device)   
        results_dict[seed]["Amnesiac Unlearning"]["Loss MIA"] =   loss_results   

        results_dict[seed]["Label Smoothing Unlearning"] = {} 
        ls_model,results_dict[seed]["Label Smoothing Unlearning"] = um.label_smoothing_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_impair,n_epoch_repair,results_dict[seed]["Label Smoothing Unlearning"],n_classes,forget_amount,architecture,seed)
        logit_distributions(ls_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'label_smoothing_loss')
        
        results_dict[seed]["Label Smoothing Unlearning"]["Activation distance"]  = unlearn_metrics.actviation_distance(ls_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Label Smoothing Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(ls_model,naive_model,forget_eval_loader,device)  
        loss_results = unlearn_metrics.mia_efficacy(ls_model,forget_loader,n_classes,device)
        results_dict[seed]["Label Smoothing Unlearning"]["Loss MIA"] =   loss_results   

        print(f'All unlearning methods applied for seed: {seed}.\n{results_dict}')

                
    with open(f"{save_dir}unlearning_results.json",'w') as f:
        json.dump(results_dict,f)
    
# Removes random instances from the dataset to form the forget and remain set
def forget_rand_datasets(dataset_pointer,pipeline,forget_percentage,device,num_classes):
    train_set,test_set = ld.load_datasets(dataset_pointer,pipeline,True)
    forget_instances_num = math.ceil(((len(train_set)/100)*forget_percentage)) 
    print("Creating remain and forget datasets")
    remain_set,forget_set = um.create_forget_remain_set(dataset_pointer,forget_instances_num,train_set)
    num_remain_set = len(remain_set)
    num_forget_set = len(forget_set)
    print(f"Remain instances: {num_remain_set}")
    print(f"Forget instances: {num_forget_set}")
    forget_randl_set = forget_set
    remain_set = ld.DatasetProcessor(remain_set,device)
    forget_set = ld.DatasetProcessor(forget_set,device)
    test_set = ld.DatasetProcessor(test_set,device)
    forget_randl_data = ld.DatasetProcessor_randl(forget_randl_set,device,num_classes)
    remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader = create_loaders(remain_set,forget_set,test_set,forget_randl_data)
    return remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,num_forget_set

# Removes random from the dataset to form the forget and remain set
def forget_class_datasets(dataset_pointer,pipeline,forget_classes_num,n_classes,device):
    train_set,test_set = ld.load_datasets(dataset_pointer,pipeline,True)
    print(f"Number of classes to remove  {forget_classes_num}")
    print("Creating remain and forget datasets")
    forget_set,remain_set,test_set = um.class_removal(dataset_pointer,forget_classes_num,n_classes,train_set,test_set)
    num_remain_set = len(remain_set)
    num_forget_set = len(forget_set)
    print(f"Remain instances: {num_remain_set}")
    print(f"Forget instances: {num_forget_set}")
    forget_randl_set = forget_set
    test_set = ld.DatasetProcessor(test_set,device)
    remain_set = ld.DatasetProcessor(remain_set,device)
    forget_set = ld.DatasetProcessor(forget_set,device)
    forget_randl_data = ld.DatasetProcessor_randl(forget_randl_set,device,n_classes)

    remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader = create_loaders(remain_set,forget_set,test_set,forget_randl_data)
    return remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,num_forget_set
    
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
            remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,forget_number = forget_rand_datasets(dataset_pointer,pipeline,forget_percentage,device,n_classes) 
        elif forget_classes == True:
            remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,forget_number = forget_class_datasets(dataset_pointer,pipeline,forget_classes_num,n_classes,device) 
        unlearning_process(remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,dataset_pointer,architecture,n_epochs,seeds,n_classes,n_inputs,n_epoch_impair,n_epoch_repair,n_epochs_fine_tune,forget_number,pruning_ratio,tag,device)
    print("FIN")

if __name__ == "__main__":
    with open("./configs/base_config.json","r") as b:
        config_base = json.load(b)    
    with open("./configs/unlearn_config.json","r") as u:
        config_unlearn = json.load(u)
    main(config_unlearn,config_base)
