import torch.optim as optim
import torch
import torch.nn as nn
import os
import vgg 
from vgg import VGGish,VGG9
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import torchmetrics.classification 
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
    criterion = nn.CrossEntropyLoss()
    return optimizer,criterion

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialise_model(architecture,n_inputs,n_classes,device,lr=0.01):
    if architecture == 'VGGish':
        model = VGGish(n_inputs,n_classes)

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
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_all = pd.DataFrame()

    # Process training set
    for batch_idx,(data,target) in enumerate(tqdm(train_loader)):
        data,target = data.to(device),target.to(device)
        with torch.no_grad():
            logits_train = model(data)
            logits_softmax = nn.Softmax(dim=1)(logits_train)
            numpy_logits_train = logits_softmax.cpu().numpy()
            df_logit_train = pd.DataFrame(numpy_logits_train)
            df_train = pd.concat([df_train,df_logit_train],ignore_index=True)
    df_train['label'] = 0

    # Process test set
    for data,target in test_loader:
        data,target = data.to(device),target.to(device)
        with torch.no_grad():
            logits_test = model(data)
            logit_test_softmax = nn.Softmax(dim=1)(logits_test)
            numpy_logits_test = logit_test_softmax.cpu().numpy()
            df_logit_test = pd.DataFrame(numpy_logits_test)
            df_test = pd.concat([df_test,df_logit_test],ignore_index=True)
    df_test['label'] = 1

    df_all = pd.concat([df_train,df_test],ignore_index=True)

    return df_all

def logits_unlearn(model,forget_loader,device):
    model.to(device)

    model.eval()
    df_forget = pd.DataFrame()

    # Process training set
    for batch_idx,(data,target) in enumerate(tqdm(forget_loader)):
        data,target = data.to(device),target.to(device)
        with torch.no_grad():
            logits_train = model(data)
            logits_softmax = nn.Softmax(dim=1)(logits_train)
            numpy_logits_forget = logits_softmax.cpu().numpy()
            df_logit_forget = pd.DataFrame(numpy_logits_forget)
            df_forget = pd.concat([df_forget,df_logit_forget],ignore_index=True)
    return df_forget

def evaluate(model,dataloader,device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
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

    with torch.no_grad():
        for data, target in test_loader:
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








     