import torch.optim as optim
import torch
import torch.nn as nn
import os
from models.vgg import VGGishMel,VGGishSpec,VGG9,VGGishMelr,VGGishSpecr
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_hyperparameters(model,lr):
    optimizer = optim.SGD(model.parameters(),lr,momentum=0.9)
    # optimizer = optim.Adam(model.parameters(),lr)
    criterion = nn.CrossEntropyLoss()
    return optimizer,criterion

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialise_model(architecture,n_inputs,n_classes,device,lr=0.01):
    if architecture == 'VGGishMel':
        model = VGGishMel(n_inputs,n_classes)
    elif architecture == 'VGGishSpec':
        model = VGGishSpec(n_inputs,n_classes)
    elif architecture == 'VGGishMelr':
        model = VGGishMelr(n_inputs,n_classes)
    elif architecture == 'VGGishSpecr':
        model = VGGishSpecr(n_inputs,n_classes)
    

    # elif architecture == 'Transformer':
    #     model  = SimpleViT(
    #         image_size = 32,
    #         patch_size = 32,
    #         num_classes = n_classes,
    #         dim = 1024,
    #         depth = 6,
    #         heads = 16,
    #         mlp_dim = 2048
    #     )

    elif architecture == 'VGG9':
        model = VGG9()

    model.to(device)
    optimizer,criterion = set_hyperparameters(model,lr) 
    return model,optimizer,criterion

def logits(model,train_loader,test_loader,device):
    model.to(device)

    model.eval()
    df_train_logits = pd.DataFrame()
    df_test_logits = pd.DataFrame()
    df_all_logits = pd.DataFrame()
    df_train_loss = pd.DataFrame()
    df_test_loss = pd.DataFrame()
    df_all_loss = pd.DataFrame()

    # Process training set
    for batch_idx,(data,target) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            logits_train = model(data)
            logits_train_softmax = F.softmax(logits_train,dim=1)
            loss = F.cross_entropy(logits_train, target,reduction ='none')
            numpy_train_loss = loss.cpu().numpy()
            train_loss = pd.DataFrame(numpy_train_loss)
            df_train_loss = pd.concat([df_train_loss,train_loss],ignore_index=True)
    
            numpy_train_logits = logits_train_softmax.cpu().numpy()
            train_logits = pd.DataFrame(numpy_train_logits)
            df_train_logits = pd.concat([df_train_logits,train_logits],ignore_index=True)

    df_train_loss['label'] = 0    
    df_train_logits['label'] = 0

    # Process test set
    for data,target in test_loader:
        with torch.no_grad():
            logits_test = model(data)
            logit_test_softmax =  F.softmax(logits_test,dim=1)
            loss = F.cross_entropy(logits_test, target,reduction ='none')
            numpy_test_loss = loss.cpu().numpy()
            test_loss = pd.DataFrame(numpy_test_loss)
            df_test_loss = pd.concat([df_test_loss,test_loss],ignore_index=True)

            numpy_logits_test = logit_test_softmax.cpu().numpy()
            df_logit_test = pd.DataFrame(numpy_logits_test)
            df_test_logits = pd.concat([df_test_logits,df_logit_test],ignore_index=True)
    df_test_logits['label'] = 1
    df_test_loss['label'] = 1  

    df_all_logits = pd.concat([df_train_logits,df_test_logits],ignore_index=True)
    df_all_loss = pd.concat([df_train_loss,df_test_loss],ignore_index=True)
    return df_all_logits,df_all_loss

def logits_unlearn(model,forget_loader,device):
    model.to(device)

    model.eval()
    df_forget_logit = pd.DataFrame()
    df_forget_loss = pd.DataFrame()

    # Process training set
    for batch_idx,(data,target) in enumerate(tqdm(forget_loader)):
        with torch.no_grad():
            logits = model(data)
            logit_softmax =  F.softmax(logits,dim=1)
            loss = F.cross_entropy(logits, target,reduction ='none')
            numpy_loss = loss.cpu().numpy()
            forget_loss = pd.DataFrame(numpy_loss)
            df_forget_loss = pd.concat([df_forget_loss,forget_loss],ignore_index=True)

            numpy_logits = logit_softmax.cpu().numpy()
            forget_logit = pd.DataFrame(numpy_logits)
            df_forget_logit = pd.concat([df_forget_logit,forget_logit],ignore_index=True)
    df_forget_loss['label'] = 1
    df_forget_logit['label'] = 1  
    return df_forget_logit,df_forget_loss

def evaluate(model,dataloader,device):
    model.eval()
    correct = 0
    total = 0
    for data, target in dataloader:
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def evaluate_test(model,test_loader,criterion,n_classes,device):
    metric = MulticlassCalibrationError(n_classes, n_bins=15, norm='l1')
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    ece = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            ece += metric(output,target).item()
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    ece /= len(test_loader)
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    return test_accuracy,test_loss, ece








     