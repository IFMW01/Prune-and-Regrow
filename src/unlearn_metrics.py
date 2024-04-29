import torch
import torch.nn.functional as F
import json
from glob import glob
import pandas as pd
from sklearn.metrics import accuracy_score
import statistics
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
    distances = []
    for batch, (data,label) in enumerate(dataloader):
        data = data.to(device)
        unlearn_outputs = unlearn_model(data)
        retrain_outputs = retrain_model(data)
        diff = torch.sqrt(torch.sum(torch.square(F.softmax(unlearn_outputs, dim = 1) - F.softmax(retrain_outputs, dim = 1)), axis = 1))
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis = 0)
    return distances.mean()

def JS_divergence(unlearn_model, retrain_model,dataloader,device):
    js_divergence = []
    for batch, (data,label) in enumerate(dataloader):
        data = data.to(device)
        unlearn_outputs = unlearn_model(data)
        retrain_outputs = retrain_model(data)
        unlearn_outputs = F.softmax(unlearn_outputs,dim=1)
        retrain_outputs = F.softmax(retrain_outputs,dim=1)
        unlearn_loss = F.cross_entropy(unlearn_outputs, label,reduction ='none')
        retrain_loss = F.cross_entropy(retrain_outputs, label,reduction ='none')
        diff = (unlearn_loss+retrain_loss)/2 
        js = (0.5*F.kl_div(torch.log(unlearn_loss), diff) + 0.5*F.kl_div(torch.log(retrain_loss), diff)).detach().cpu().item()
        js_divergence.append(js)
    return statistics.mean(js_divergence)


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
        
    return logits_dict,loss_dict



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
            acc = accuracy_score(labels.values, y_pred)
            if key in output_dictionary:
                output_dictionary[key].append(acc)
            else:
                output_dictionary[key] = [acc]
    return output_dictionary 





