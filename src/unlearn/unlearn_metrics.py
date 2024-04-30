import torch
import torch.nn.functional as F
import json
import glob
import pandas as pd
from sklearn.metrics import accuracy_score
import statistics
import utils
import torch.nn as nn
from models.attack_model import SoftmaxModel
from torch.utils.data import DataLoader

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
    return distances.mean().item()

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


def mia_efficacy(model,forget_loader,n_classes,device):
    df_forget_logit,df_forget_loss = utils.logits_unlearn(model,forget_loader,device)
    attack_model_list_logit =  glob.glob(f'TRAIN/{dataset_pointer}/{architecture}/MIA/Logits/Attack/*.pth')
    attack_model_list_loss =  glob.glob(f'TRAIN/{dataset_pointer}/{architecture}/MIA/Loss/Attack/*.pth')
    logits_results =  attack_results(attack_model_list_logit,n_classes,df_forget_logit)
    loss_results = attack_results(attack_model_list_loss,1,df_forget_loss)
    return logits_results,loss_results

def attack_results(model_list,n_inputs,df):
    attack_sucess = []
    labels = df['label']
    df = df.drop(['label'],axis=1)

    x_train = torch.tensor(df.values,dtype=torch.float)
    y_train = torch.tensor(labels.values,dtype=torch.long)
    forget_set = [(x_train[i], y_train[i]) for i in range(len(x_train))]
    forget_laoder = DataLoader(forget_set, batch_size=264, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for attack_path in model_list:
        attack_model = torch.load(attack_path)
        attack_model.eval()
        model_loss = 0.0
        correct = 0
        total = 0

        for data, target in forget_laoder:
            with torch.no_grad():
                # data = data.to(self.device)
                # target = target.to(self.device)
                output = attack_model(data)
                loss = criterion(output, target)
                model_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        model_loss /= len(forget_laoder)
        accuracy = 100 * correct / total
        attack_sucess.append(accuracy)
    return attack_sucess 





