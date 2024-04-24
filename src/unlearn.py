import json
import os
import unlearning_methods as um
import load_datasets as ld
import glob
import utils 
import math
import random
import numpy as np
import unlearn_metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from load_datasets import DatasetProcessor,DatasetProcessor_randl


def unlearn_logits(model,loader,device,save_dir,filename_logits,filename_loss):
    logits,loss = utils.logits_unlearn(model,loader,device)
    logits.to_csv(f'{save_dir}{filename_logits}.csv',index = False)
    loss.to_csv(f'{save_dir}{filename_loss}.csv',index = False)

def logit_distributions(model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,filename_logits,filename_loss):
   unlearn_logits(model,remain_eval_loader,device,save_dir,f'{filename_logits}_remain',f'{filename_loss}_remain')
   unlearn_logits(model,forget_eval_loader,device,save_dir,f'{filename_logits}_forget',f'{filename_loss}_forget')
   unlearn_logits(model,test_loader,device,save_dir,f'{filename_logits}_test',f'{filename_loss}_test')
     

def randomise_lables(data_set,dataset_pointer):
    if dataset_pointer == 'SpeechCommands':
        labels = np.load('./labels/speech_commands_labels.npy')
        labels = labels.tolist()
        for i in range(len(data_set)):
          label_index = labels.index(data_set[i][2])
          current_label = label_index
          while current_label == label_index:
              current_label = random.randint(0, (len(labels)-1))
          data_set[i][2] = labels[current_label]
    elif dataset_pointer == 'CIFAR10':
        labels = np.load('./labels/cifar10_labels.npy')
        labels = labels.tolist()
        for i in range(len(data_set)):
          label = data_set[i][1]
          current_label = label
          while current_label == label:
              current_label = random.randint(0, (len(labels)-1))  
          data_set[i][1] = current_label    
    return data_set

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
    forget_percentage = config_unlearn.get("forget_percentage",None)
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
    print(f"Percentage of data to forget: {forget_percentage}")
    print(f"Pruning ratio: {pruning_ratio}")

    device = utils.get_device()
    model_dir = f'TRAIN/{dataset_pointer}/{architecture}'
    if not os.path.exists(model_dir):
        print(f"There are no models with this {architecture} for this {dataset_pointer} in the TRAIN directory. Please train relevant models")
        return
    else:
        train_set,test_set = ld.load_datasets(dataset_pointer,pipeline,True)
        forget_instances_num = math.ceil(((len(train_set)/100)*forget_percentage)) 
        remain_set,forget_set = um.create_forget_remain_set(dataset_pointer,forget_instances_num,train_set)
        print(f"Remain instances: {len(remain_set)}")
        print(f"Forget instances: {len(forget_set)}")
        print("Creating remain and forget data loaders")
        if dataset_pointer == 'SpeechCommands':
          remain_loader = ld.trainset_loader(remain_set)
          remain_eval_loader = ld.testset_loader(remain_set)
          forget_loader = ld.trainset_loader(forget_set)
          forget_eval_loader =  ld.testset_loader(forget_set)
          forget_randl_data = randomise_lables(forget_set,dataset_pointer)
          forget_randl_loader = ld.trainset_loader(forget_randl_data)
          test_loader = ld.testset_loader(test_set)
        elif dataset_pointer == 'CIFAR10':
          forget_randl_data = randomise_lables(forget_set,dataset_pointer)
        elif dataset_pointer == 'audioMNIST' or  dataset_pointer == 'Ravdess':
          forget_randl_set = forget_set
          remain_set = DatasetProcessor(remain_set)
          forget_set = DatasetProcessor(forget_set)
          test_set = DatasetProcessor(test_set)
          forget_randl_data = DatasetProcessor_randl(forget_randl_set)
        
        if dataset_pointer == 'CIFAR10' or dataset_pointer == 'audioMNIST'  or dataset_pointer == 'Ravdess':
            remain_loader = DataLoader(remain_set, batch_size=256,
                                                shuffle=True, num_workers=2)
            remain_eval_loader = DataLoader(remain_set, batch_size=256,
                                                shuffle=False, num_workers=2)
            forget_loader = DataLoader(forget_set, batch_size=256,
                                                shuffle=True, num_workers=2)
            forget_eval_loader = DataLoader(forget_set, batch_size=256,
                                                shuffle=False, num_workers=2)       
            test_loader = DataLoader(test_set, batch_size=256,
                                                shuffle=False, num_workers=2)
            forget_randl_loader = DataLoader(forget_randl_data, batch_size=256,
                                                shuffle=True, num_workers=2)
        
        results_dict = {}

        for seed in seeds:
            results_dict[seed] = {}
            model_dir = f'TRAIN/{dataset_pointer}/{architecture}/{seed}'
            save_dir = f"TRAIN/{dataset_pointer}/{architecture}/UNLEARN/{forget_percentage}/{seed}/"
            utils.create_dir(save_dir)
            print(f"Acessing trained model on seed: {seed}")
            model_path = glob.glob(os.path.join(model_dir,'*.pth'))
            model_path = model_path[0]
                   
            orginal_model,optimizer,criterion = um.load_model(model_path,0.01,device)
            logit_distributions(orginal_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'orginal_model_logits','orginal_model_loss')

            naive_model,results_dict[seed] = um.naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epochs,results_dict[seed],seed)
            logit_distributions(naive_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'naive_model_logits','naive_model_loss')
            results_dict[seed]["Original Model"] = []
            results_dict[seed]["Original Model"].append(unlearn_metrics.actviation_distance(orginal_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Original Model"].append(unlearn_metrics.JS_divergence(orginal_model,naive_model,forget_eval_loader,device)) 

            results_dict[seed]["Naive Unlearning"].append(unlearn_metrics.actviation_distance(naive_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Naive Unlearning"].append(unlearn_metrics.JS_divergence(naive_model,naive_model,forget_eval_loader,device))      


            gradient_ascent_model,results_dict[seed] = um.gradient_ascent(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,device,n_epoch_impair,n_epoch_repair,results_dict[seed],n_classes,forget_instances_num,dataset_pointer,seed)
            logit_distributions(gradient_ascent_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'gradient_ascent_model_logits','gradient_ascent_model_loss')
            results_dict[seed]["Gradient Ascent Unlearning"].append(unlearn_metrics.actviation_distance(gradient_ascent_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Gradient Ascent Unlearning"].append(unlearn_metrics.JS_divergence(gradient_ascent_model,naive_model,forget_eval_loader,device))      

            fine_tuning_model,results_dict[seed] = um.fine_tuning_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epochs_fine_tune,results_dict[seed],n_classes,seed)
            logit_distributions(fine_tuning_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'fine_tuning_model_logits','fine_tuning_model_loss')
            results_dict[seed]["Fine Tune Unlearning"].append(unlearn_metrics.actviation_distance(fine_tuning_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Fine Tune Unlearning"].append(unlearn_metrics.JS_divergence(fine_tuning_model,naive_model,forget_eval_loader,device))       

            stochastic_teacher_model,results_dict[seed] = um.stochastic_teacher_unlearning(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,device,n_inputs,n_classes,architecture,results_dict[seed],n_epoch_impair,n_epoch_repair,seed)
            logit_distributions(stochastic_teacher_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'stochastic_teacher_model_logits','stochastic_teacher_model_loss')
            results_dict[seed]["Stochastic Teacher Unlearning"].append(unlearn_metrics.actviation_distance(stochastic_teacher_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Stochastic Teacher Unlearning"].append(unlearn_metrics.JS_divergence(stochastic_teacher_model,naive_model,forget_eval_loader,device))       

            omp_model,results_dict[seed] = um. omp_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,pruning_ratio,n_epochs_fine_tune,results_dict[seed],n_classes,seed)
            logit_distributions(omp_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'omp_model_logits','omp_model_loss')
            results_dict[seed]["OMP Unlearning"].append(unlearn_metrics.actviation_distance(omp_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["OMP Unlearning"].append(unlearn_metrics.JS_divergence(omp_model,naive_model,forget_eval_loader,device))       


            cosine_model,results_dict[seed] = um.cosine_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epochs_fine_tune,results_dict[seed],n_classes,seed)
            logit_distributions(cosine_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'cosine_model_logits','cosine_model_loss')
            results_dict[seed]["Cosine Unlearning"].append(unlearn_metrics.actviation_distance(cosine_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Cosine Unlearning"].append(unlearn_metrics.JS_divergence(cosine_model,naive_model,forget_eval_loader,device))       


            kk_model,results_dict[seed] = um.kurtosis_of_kurtoses_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epochs_fine_tune,results_dict[seed],n_classes,seed)
            logit_distributions(kk_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'kk_model_logits','kk_model_loss')
            results_dict[seed]["Kurtosis Unlearning"].append(unlearn_metrics.actviation_distance(kk_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Kurtosis Unlearning"].append(unlearn_metrics.JS_divergence(kk_model,naive_model,forget_eval_loader,device))    


            randl_model,results_dict[seed] = um.randl_unlearning(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,forget_randl_loader,device,n_epoch_impair,n_epoch_repair,results_dict[seed],n_classes,seed)
            logit_distributions(randl_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'randl_model_logits','randl_model_loss')
            results_dict[seed]["Amnesiac Unlearning"].append(unlearn_metrics.actviation_distance(randl_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Amnesiac Unlearning"].append(unlearn_metrics.JS_divergence(randl_model,naive_model,forget_eval_loader,device))    


            ls_model,results_dict[seed] = um.label_smoothing_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_impair,n_epoch_repair,results_dict[seed],n_classes,seed)
            logit_distributions(ls_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'label_smoothing_logits','label_smoothing_loss')
            results_dict[seed]["Label Smoothing Unlearning"].append(unlearn_metrics.actviation_distance(ls_model, naive_model, forget_eval_loader, device))
            results_dict[seed]["Label Smoothing Unlearning"].append(unlearn_metrics.JS_divergence(ls_model,naive_model,forget_eval_loader,device))    

            print(f'All unlearning methods applied for seed: {seed}.\n{results_dict}')
  
        with open(f"{save_dir}/unlearning_results.json",'w') as f:
            json.dump(results_dict,f)
        # unlearn_metrics.mia_efficacy()  
        print("FIN")

if __name__ == "__main__":
    with open("./configs/base_config.json","r") as b:
        config_base = json.load(b)    
    with open("./configs/unlearn_config.json","r") as u:
        config_unlearn = json.load(u)
    main(config_unlearn,config_base)
