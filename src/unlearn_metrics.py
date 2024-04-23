import torch.nn as nn 
import torch
import torch.functional as F
import utils
import json
from glob import glob
import pandas as pd
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier

with open("./configs/base_config.json","r") as b:
    config_base = json.load(b)    
with open("./configs/attack_config.json", "r") as a:
    config_attack = json.load(a)
with open("./configs/unlearn_config.json","r") as u:
    config_unlearn = json.load(u)


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


def actviation_distance(unlearn_model, retrain_model, dataloader, device):
    sftmx = nn.Softmax(dim = 1)
    distances = []
    for batch in dataloader:
        x, _, _ = batch
        x = x.to(device)
        unlearn_outputs = unlearn_model(x)
        retrain_outputs = retrain_model(x)
        diff = torch.sqrt(torch.sum(torch.square(F.softmax(unlearn_outputs, dim = 1) - F.softmax(retrain_outputs, dim = 1)), axis = 1))
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis = 0)
    return distances.mean()

def JS_divergence(unlearn_model, retrain_model,forget_eval_loader,device):
    df_unlearn_logit,df_unlearn_loss = utils.logits_unlearn((unlearn_model,forget_eval_loader,device))
    df_retrain_logit,df_retrain_loss = utils.logits_unlearn((retrain_model,forget_eval_loader,device))
    diff = (df_unlearn_loss+df_retrain_loss)/2
    js_divergence = 0.5*F.kl_div(torch.lxog(df_unlearn_loss), diff) + 0.5*F.kl_div(torch.log(df_retrain_loss), diff)
    return js_divergence

def mia_efficacy():
    logits_dict = {}
    loss_dict = {}
    for seed in seeds:
        unlearn_dir = f"TRAIN/{dataset_pointer}/{architecture}/UNLEARN/{forget_percentage}/{seed}/"
        logits_list = glob.glob(f'{unlearn_dir} *logits_forget.csv')
        loss_list = glob.glob(f'{unlearn_dir} *loss_forget.csv')
        attack_model_list =  glob.glob(f'TRAIN/{dataset_pointer}/{architecture}/MIA/Loss')
        logits_dict[seed] = attack_results(attack_model_list,logits_list)
        loss_dict[seed] = attack_results(attack_model_list,loss_list)
        
    save_dict_path = f"TRAIN/{dataset_pointer}/{architecture}/UNLEARN/{forget_percentage}/"
    with open(f"{save_dict_path}_mia_efficacy_logits",'w') as f:
        json.dump(logits_dict,f)
    with open(f"{save_dict_path}_mia_efficacy_loss",'w') as f:
        json.dump(loss_dict,f)


def attack_results(model_list,unlearn_list):
    output_dictionary = {}
    for attack_path in model_list:
        attack_model = TabNetClassifier()
        attack_model.load_model(attack_path)
        for method in unlearn_list:
            key = method.split('_')[0]
            df = pd.read_csv(method)
            labels = df['label']
            df = df.drop(['label'],axis=1)
            y_pred = attack_model(df)
            loss_mia_acc = accuracy_score(labels.values, y_pred)
            if key in output_dictionary:
                output_dictionary[key].append(loss_mia_acc)
            else:
                output_dictionary[key] = [loss_mia_acc]
    return output_dictionary 





