import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data import DataLoader
import json
import os
import glob
import utils 
import math
import os.path
from unlearn import unlearning_methods as um
from unlearn import unlearn_metrics
from datasets_unlearn import load_datasets as ld
import numpy as np
import random
import argparse

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    generator = torch.Generator()
    generator.manual_seed(0)
    remain_loader = DataLoader(remain_set, batch_size=256,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=generator)
    remain_eval_loader = DataLoader(remain_set, batch_size=256,
                                    shuffle=False,
                                    worker_init_fn=seed_worker,
                                    generator=generator)
    forget_loader = DataLoader(forget_set, batch_size=256,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=generator)
    forget_eval_loader = DataLoader(forget_set, batch_size=256,
                                    shuffle=False,
                                    worker_init_fn=seed_worker,
                                    generator=generator)       
    test_loader = DataLoader(test_set, batch_size=256,
                            shuffle=False,
                            worker_init_fn=seed_worker,
                            generator=generator)
    forget_randl_loader = DataLoader(forget_randl_data, batch_size=256,
                                    shuffle=True,
                                    worker_init_fn=seed_worker,
                                    generator=generator)
    return remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader


# Calls all unlearning methods to be perfromed on the base models that have been created and saves the results
def unlearning_process(remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,dataset_pointer,architecture,n_epochs,seed,n_classes,n_inputs,n_epoch_impair,n_epoch_repair,forget_amount,tag,device):
            
    results_dict = {}

    print(f'\nSeed: {seed}')
    results_dict[seed] = {}
    model_dir = f'Results/{dataset_pointer}/{architecture}/{seed}'
    save_dir = f"Results/{dataset_pointer}/{architecture}/UNLEARN/{tag}/{forget_amount}/{seed}/"
    utils.create_dir(save_dir)
    print(f"Acessing trained model on seed: {seed}")
    model_path = glob.glob(os.path.join(model_dir,'*.pth'))
    model_path = model_path[0]
            
    orginal_model,optimizer,criterion = um.load_model(model_path,0.01,device)
    results_dict[seed]["Original Model"] = {}
    logit_distributions(orginal_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'orginal_model_loss')

    results_dict[seed]["Naive Unlearning"] = {}
    if not os.path.isfile(f"{save_dir}Naive.pth"):
        naive_model,results_dict[seed]["Naive Unlearning"] = um.naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epochs,results_dict[seed]["Naive Unlearning"],seed)
        logit_distributions(naive_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'naive_model_loss')
        torch.save(naive_model,f"{save_dir}Naive.pth")
        results_dict[seed]["Original Model"]["Activation distance"] = unlearn_metrics.actviation_distance(orginal_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Original Model"]["JS divergance"]  = unlearn_metrics.JS_divergence(orginal_model,naive_model,forget_eval_loader,device)
        loss_results = unlearn_metrics.mia_efficacy(orginal_model,forget_loader,n_classes,dataset_pointer,architecture,device)
        results_dict[seed]["Original Model"]["Loss MIA"] = loss_results

        results_dict[seed]["Naive Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(naive_model, naive_model, forget_eval_loader, device)
        results_dict[seed]["Naive Unlearning"]["JS divergance"] = (unlearn_metrics.JS_divergence(naive_model,naive_model,forget_eval_loader,device)) 
        loss_results = unlearn_metrics.mia_efficacy(naive_model,forget_loader,n_classes,dataset_pointer,architecture,device)
        results_dict[seed]["Naive Unlearning"]["Loss MIA"] =   loss_results 
        with open(f"{save_dir}Niave.json",'w') as f:
            json.dump(results_dict,f)
    else:
        naive_model,optimizer,criterion = um.load_model(f"{save_dir}Naive.pth",0.01,device)
    
    results_dict[seed]["Gradient Ascent Unlearning"] = {}
    gradient_ascent_model,results_dict[seed]["Gradient Ascent Unlearning"] = um.gradient_ascent(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,device,n_epoch_impair,n_epoch_repair,results_dict[seed]["Gradient Ascent Unlearning"],n_classes,forget_amount,dataset_pointer,architecture,seed)
    logit_distributions(gradient_ascent_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'gradient_ascent_model_loss')
    results_dict[seed]["Gradient Ascent Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(gradient_ascent_model, naive_model, forget_eval_loader, device)
    results_dict[seed]["Gradient Ascent Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(gradient_ascent_model,naive_model,forget_eval_loader,device)

    loss_results = unlearn_metrics.mia_efficacy(gradient_ascent_model,forget_loader,n_classes,dataset_pointer,architecture,device)
    results_dict[seed]["Gradient Ascent Unlearning"]["Loss MIA"] =   loss_results    

    results_dict[seed]["Fine Tune Unlearning"] = {}
    fine_tuning_model,results_dict[seed]["Fine Tune Unlearning"] = um.fine_tuning_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_repair,results_dict[seed]["Fine Tune Unlearning"],n_classes,architecture,seed)
    logit_distributions(fine_tuning_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'fine_tuning_model_loss')

    results_dict[seed]["Fine Tune Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(fine_tuning_model, naive_model, forget_eval_loader, device)
    results_dict[seed]["Fine Tune Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(fine_tuning_model,naive_model,forget_eval_loader,device)

    loss_results = unlearn_metrics.mia_efficacy(fine_tuning_model,forget_loader,n_classes,dataset_pointer,architecture,device)
    results_dict[seed]["Fine Tune Unlearning"]["Loss MIA"] =   loss_results    

    results_dict[seed]["Stochastic Teacher Unlearning"] = {}
    stochastic_teacher_model,results_dict[seed]["Stochastic Teacher Unlearning"]= um.stochastic_teacher_unlearning(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,device,n_inputs,n_classes,architecture,results_dict[seed]["Stochastic Teacher Unlearning"],n_epoch_impair,n_epoch_repair,seed)
    logit_distributions(stochastic_teacher_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'stochastic_teacher_model_loss')
    results_dict[seed]["Stochastic Teacher Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(stochastic_teacher_model, naive_model, forget_eval_loader, device)
    results_dict[seed]["Stochastic Teacher Unlearning"]["JS divergance"]= unlearn_metrics.JS_divergence(stochastic_teacher_model,naive_model,forget_eval_loader,device)     

    loss_results = unlearn_metrics.mia_efficacy(stochastic_teacher_model,forget_loader,n_classes,dataset_pointer,architecture,device)
    results_dict[seed]["Stochastic Teacher Unlearning"]["Loss MIA"] =   loss_results   

    results_dict[seed]["Amnesiac Unlearning"] = {} 
    amnesiac_model,results_dict[seed]["Amnesiac Unlearning"] = um.amnesiac_unlearning(model_path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,forget_randl_loader,device,n_epoch_impair,n_epoch_repair,results_dict[seed]["Amnesiac Unlearning"],n_classes,architecture,seed)
    logit_distributions(amnesiac_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'amnesiac_model_loss')

    results_dict[seed]["Amnesiac Unlearning"]["Activation distance"]  = unlearn_metrics.actviation_distance(amnesiac_model, naive_model, forget_eval_loader, device)
    results_dict[seed]["Amnesiac Unlearning"]["JS divergance"]  = unlearn_metrics.JS_divergence(amnesiac_model,naive_model,forget_eval_loader,device)
    loss_results = unlearn_metrics.mia_efficacy(amnesiac_model,forget_loader,n_classes,dataset_pointer,architecture,device)   
    results_dict[seed]["Amnesiac Unlearning"]["Loss MIA"] =   loss_results   

    results_dict[seed]["OMP Unlearning"] = {}
                                                    
    omp_model,results_dict[seed]["OMP Unlearning"] = um.omp_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_repair,results_dict[seed]["OMP Unlearning"],n_classes,architecture,seed)
    logit_distributions(omp_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'omp_model_loss')

    results_dict[seed]["OMP Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(omp_model, naive_model, forget_eval_loader, device)
    results_dict[seed]["OMP Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(omp_model,naive_model,forget_eval_loader,device)     

    loss_results = unlearn_metrics.mia_efficacy(omp_model,forget_loader,n_classes,dataset_pointer,architecture,device)   
    results_dict[seed]["OMP Unlearning"]["Loss MIA"] =   loss_results    

    results_dict[seed]["Cosine Unlearning"] = {} 
    cosine_model,results_dict[seed]["Cosine Unlearning"] = um.cosine_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_repair,results_dict[seed]["Cosine Unlearning"],n_classes,architecture,n_inputs,seed)
    logit_distributions(cosine_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'cosine_model_loss')
    
    results_dict[seed]["Cosine Unlearning"]["Activation distance"] = unlearn_metrics.actviation_distance(cosine_model, naive_model, forget_eval_loader, device)
    results_dict[seed]["Cosine Unlearning"]["JS divergance"] = unlearn_metrics.JS_divergence(cosine_model,naive_model,forget_eval_loader,device)  
    loss_results = unlearn_metrics.mia_efficacy(cosine_model,forget_loader,n_classes,dataset_pointer,architecture,device)    
    results_dict[seed]["Cosine Unlearning"]["Loss MIA"] =   loss_results    

    results_dict[seed]["POP"] = {} 
    pop_model,results_dict[seed]["POP"] = um.pop_unlearning(model_path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_eval_loader,n_epoch_repair,results_dict[seed]["POP"],n_classes,architecture,n_inputs,seed)
    logit_distributions(pop_model,remain_eval_loader,forget_eval_loader,test_loader,device,save_dir,'pop_model_loss')

    results_dict[seed]["POP"]["Activation distance"]  = unlearn_metrics.actviation_distance(pop_model, naive_model, forget_eval_loader, device)
    results_dict[seed]["POP"]["JS divergance"]  = unlearn_metrics.JS_divergence(pop_model,naive_model,forget_eval_loader,device)
    loss_results = unlearn_metrics.mia_efficacy(pop_model,forget_loader,n_classes,dataset_pointer,architecture,device) 
    results_dict[seed]["POP"]["Loss MIA"] =   loss_results   

    print(f'All unlearning methods applied for seed: {seed}.\n{results_dict}')

    with open(f"{save_dir}unlearning_results.json",'w') as f:
        json.dump(results_dict,f)
    
# Removes random instances from the dataset to form the forget and remain set
def forget_rand_datasets(dataset_pointer,pipeline,forget_percentage,device,num_classes):
    train_set,test_set = ld.load_datasets(dataset_pointer,pipeline,True)
    forget_instances_num = math.ceil(((len(train_set)/100)*forget_percentage)) 
    print("Creating remain and forget datasets")
    print(len(train_set))
    remain_set,forget_set = um.create_forget_remain_set(dataset_pointer,forget_instances_num,train_set)
    num_remain_set = len(remain_set)
    num_forget_set = len(forget_set)
    print(f"Remain instances: {num_remain_set}")
    print(f"Forget instances: {num_forget_set}")
    forget_randl_set = forget_set
    if dataset_pointer == 'SpeechCommands' or dataset_pointer == 'audioMNIST' or dataset_pointer == 'Ravdess' or dataset_pointer == 'UrbanSound8K':
        remain_set = ld.DatasetProcessor(remain_set,device)
        forget_set = ld.DatasetProcessor(forget_set,device)
        test_set = ld.DatasetProcessor(test_set,device)
        forget_randl_data = ld.DatasetProcessor_randl(forget_randl_set,device,num_classes)
    elif dataset_pointer == 'CIFAR10' or dataset_pointer == 'CIFAR100':
        forget_randl_data = ld.DatasetProcessor_randl_cifar(forget_randl_set,device,num_classes)
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
    if dataset_pointer == 'SpeechCommands' or dataset_pointer == 'audioMNIST' or dataset_pointer == 'Ravdess' or dataset_pointer =='UrbanSound8K':
        remain_set = ld.DatasetProcessor(remain_set,device)
        forget_set = ld.DatasetProcessor(forget_set,device)
        test_set = ld.DatasetProcessor(test_set,device)
        forget_randl_data = ld.DatasetProcessor_randl(forget_randl_set,device,n_classes)
    elif dataset_pointer == 'CIFAR10' or dataset_pointer == 'CIFAR100':
        forget_randl_data = ld.DatasetProcessor_randl_cifar(forget_randl_set,device,n_classes)

    remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader = create_loaders(remain_set,forget_set,test_set,forget_randl_data)
    return remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,num_forget_set

def options_parser():
    parser = argparse.ArgumentParser(description="Arguments for creating model")
    parser.add_argument(
        "--dataset_pointer",
        required=True,
        type=str
    )
    parser.add_argument(
        "--pipeline",
        required=False,
        default='mel',
        type=str,
    )
    parser.add_argument(
        "--architecture",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--n_epochs",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--n_classes", 
        required=True, 
        type=int, 
    )
    parser.add_argument(
        "--n_inputs",
        required=False,
        default=1,
        type=int,
        help="This is only used for audio which is mono",
    )

    parser.add_argument(
        "--unlearning",
        required=False,
        default=True,
        type=bool,
    )

    parser.add_argument(
        "--n_epoch_impair",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--n_epoch_repair",
        required=True,
        type=int,
    )


    parser.add_argument(
        "--forget_random",
        required=False,
        type=bool,
    )

    parser.add_argument(
        "--forget_percentage",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--forget_classes",
        required=False,
        type=bool,
    )

    parser.add_argument(
        "--forget_classes_num",
        required=False,
        type=int,
    )

    args = parser.parse_args()

    return args

    
def main(args):
    dataset_pointer = args.dataset_pointer
    pipeline = args.pipeline
    architecture = args.architecture
    n_epochs =args.n_epochs
    seed = args.seed
    n_classes = args.n_classes
    n_inputs = args.n_inputs
    unlearning = args.unlearning
    n_epoch_impair = args.n_epoch_impair
    n_epoch_repair = args.n_epoch_repair
    forget_random = args.forget_random
    forget_percentage = args.forget_percentage
    forget_classes = args.forget_classes
    forget_classes_num = args.forget_classes_num
    
    print("Received arguments from config file:")
    print(f"Unlearning: {unlearning}")
    print(f"Dataset pointer: {dataset_pointer}")
    print(f"pipeline: {pipeline}")
    print(f"Architecture: {architecture}")
    print(f"Number of retrain epochs: {n_epochs}")
    print(f"Seeds: {seed}")
    print(f"Number of impair epochs: {n_epoch_impair}")
    print(f"Number of repair epochs: {n_epoch_repair}")
    print(f"Forgetting random samples : {forget_random}")
    if forget_random == True:
        tag = 'Item_Removal'
        print(f"Percentage of data to forget: {forget_percentage}")
    print(f"Forgetting classes : {forget_classes}")
    if  forget_classes == True:
        tag = 'Class_Removal'
        print(f"Number of classes to forget: {forget_classes_num}")
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
        unlearning_process(remain_loader,remain_eval_loader,forget_loader,forget_eval_loader,test_loader,forget_randl_loader,dataset_pointer,architecture,n_epochs,seed,n_classes,n_inputs,n_epoch_impair,n_epoch_repair,forget_number,tag,device)
    print("FIN")

if __name__ == "__main__":
    args = options_parser()   
    main(args)
