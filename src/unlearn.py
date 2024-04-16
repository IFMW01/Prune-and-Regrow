import json
import os
import unlearning_methods as um
import load_datasets as ld
import glob
import utils 
import math
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from processAudioMNIST import AudioMNISTDataset
import processAudioMNIST as AudioMNIST

def unlearn_logits(model,forget_loader,device,save_dir,filename):
    logits = utils.logits_unlearn(model,forget_loader,device)
    logits.to_csv(f'{save_dir}{filename}.csv',index = False)

def randomise_lables(data_set,dataset_pointer):
    if dataset_pointer == 'SpeechCommands':
        lables = np.load('src/labels/speech_commands_labels.npy')
    elif dataset_pointer == 'audioMNIST':
        lables = np.load('src/labels/speech_commands_labels.npy')
    labels = labels.tolist()
    for i in range(len(data_set)):
        label_index = lables.index(data_set[i][[(len(data_set[i])-1)]])
        current_label = label_index
        while current_label == label_index:
            current_label = random.randint(0, len(lables))
        data_set[i][(len(data_set[i])-1)] = lables[current_label]
    
    return data_set

def main(config):
    dataset_pointer = config.get("dataset_pointer",None)
    pipeline = config.get("pipeline",None)
    architecture = config.get("architecture",None)
    n_epochs = config.get("n_epochs",None)
    seeds = config.get("seeds",None)
    n_classes = config.get("n_classes",None)
    n_inputs = config.get("n_inputs",None)
    unlearning = config.get("unlearning",None)
    n_epoch_impair = config.get("n_epoch_impair",None)
    n_epoch_repair = config.get("n_epoch_repair",None)
    n_epochs_fine_tune = config.get("n_epochs_fine_tune",None)
    forget_percentage = config.get("forget_percentage",None)
    pruning_ratio = config.get("pruning_ratio",None)
    
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
        remain_set,forget_set = um.create_forget_remain_set(forget_instances_num,train_set)
        print(f"len remain: {len(remain_set)}")
        print(f"len remain: {len(forget_set)}")
        
        print("Creating remain and forget data loaders")
        if dataset_pointer == 'SpeechCommands':
          remain_loader = ld.train_loader(remain_set,dataset_pointer)
          remain_eval_loader = ld.test_loader(remain_set,dataset_pointer)
          forget_loader = ld.train_loader(forget_set,dataset_pointer)
          forget_rand_lables = randomise_lables(forget_set,dataset_pointer)
          forget_rand_lables_loader = ld.train_loader(forget_rand_lables,dataset_pointer)
          test_loader = ld.test_loader(test_set,dataset_pointer)
        elif dataset_pointer == 'audioMNIST':
          remain_data = AudioMNISTDataset(remain_set)
          forget_data = AudioMNISTDataset(forget_set)
          test_data = AudioMNISTDataset(test_set)
          remain_loader = DataLoader(remain_data, batch_size=256, shuffle=True, num_workers=2)
          remain_eval_loader = DataLoader(remain_data, batch_size=256, shuffle=True, num_workers=2)
          forget_loader = DataLoader(forget_data, batch_size=256, shuffle=True, num_workers=2)
          test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

        results_dict = {}

        for seed in seeds:
    
            model_dir = f'TRAIN/{dataset_pointer}/{architecture}/{seed}'
            save_dir = f"TRAIN/{dataset_pointer}/{architecture}/UNLEARN/{forget_percentage}/{seed}/"
            utils.create_dir(save_dir)
            print(f"Acessing trained model on seed: {seed}")
            model_path = glob.glob(os.path.join(model_dir,'*.pth'))
            model_path = model_path[0]
            
            orginal_model,optimizer,criterion = um.load_model(model_path,0.01,device)
            unlearn_logits(orginal_model,forget_loader,device,save_dir,'orginal_model')

            naive_model,results_dict = um.naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs,results_dict,forget_instances_num,seed)
            unlearn_logits(naive_model,forget_loader,device,save_dir,'naive_model')

            gradient_ascent_model,results_dict = um.gradient_ascent(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,device,n_epoch_impair,n_epoch_repair,results_dict,n_classes,seed)
            unlearn_logits(gradient_ascent_model,forget_loader,device,save_dir,'gradient_ascent_model')

            fine_tuning_model,results_dict = um.fine_tuning_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs_fine_tune,results_dict,n_classes,seed)
            unlearn_logits(fine_tuning_model,forget_loader,device,save_dir,'fine_tuning_model')

            stochastic_teacher_model,results_dict = um.stochastic_teacher_unlearning(model_path,remain_loader,test_loader,forget_loader,device,n_inputs,n_classes,architecture,results_dict,n_epoch_impair,n_epoch_repair,seed)
            unlearn_logits(stochastic_teacher_model,forget_loader,device,save_dir,'stochastic_teacher_model')

            omp_model,results_dict = um. omp_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,pruning_ratio,n_epochs_fine_tune,results_dict,n_classes,seed)
            unlearn_logits(omp_model,forget_loader,device,save_dir,'omp_model')

            cosine_model,results_dict = um.cosine_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs_fine_tune,results_dict,n_classes,seed)
            unlearn_logits(cosine_model,forget_loader,device,save_dir,'cosine_model')

            kk_model,results_dict = um.kurtosis_of_kurtoses_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs_fine_tune,results_dict,n_classes,seed)
            unlearn_logits(kk_model,forget_loader,device,save_dir,'kk_model')

            randl_model,results_dict = um.randl_unlearning(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_rand_lables_loader,device,n_epoch_impair,n_epoch_repair,results_dict,n_classes,seed)
            unlearn_logits(randl_model,forget_loader,device,save_dir,'randl_model')

            print(f'All unlearning methods applied for seed: {seed}.\n{results_dict}')
            with open(f"{save_dir}/unlearning_results.json",'w') as f:
                json.dump(results_dict,f)
        print("FIN")

if __name__ == "__main__":
    with open("./configs/unlearn_config.json","r") as f:
        config = json.load(f)
    main(config)
