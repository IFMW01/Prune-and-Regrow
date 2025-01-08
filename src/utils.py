import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from models.vgg import make_vgg
from models.transformer import ViTcifar
from models.compact_ViT import CCT
from models.resnet import ResNet18
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from torchmetrics.classification import MulticlassCalibrationError

# Utility functions used throught the sctipt

# Updates the result dict
def update_dict(dict,best_time,best_epoch,train_accuracy,train_loss,train_ece,test_acc,test_loss,test_ece):
    dict['Train time'] = best_time
    dict['Best epoch'] = best_epoch
    dict['Train accuracy'] = train_accuracy
    dict['Train loss'] = train_loss
    dict['Train ece'] = train_ece
    dict['Test accuracy'] = test_acc
    dict['Test loss'] = test_loss
    dict['Test ece'] = test_ece
    return dict

# Sets the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Counts the number of parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Sets model hyperparameters
def set_hyperparameters(model,opt,lr):
    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr) 
    criterion = nn.CrossEntropyLoss()
    return optimizer,criterion

# Gets the device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Creates directory
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)

# Gets the intailised model for each architetcure 
def initialise_model(architecture,opt,n_classes,device,lr=0.001): 
    if architecture == 'VGG16':
        model = make_vgg('VGG16',n_classes)
    if architecture =='ResNet18':
        model = ResNet18(n_classes)
    elif architecture == 'CCTcifar':
        model = CCT(
            img_size = (32, 32),
            embedding_dim = 256,
            n_conv_layers = 2,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 6,
            num_heads = 4,
            mlp_ratio = 2.,
            num_classes = n_classes,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none'],
            n_input_channels=3,
        )
    elif architecture == 'ViTcifar':
        model = ViTcifar(num_classes = n_classes, dim = 512, depth = 6, heads = 6, mlp_dim = 1024)
    model = model.to(device)
    optimizer,criterion = set_hyperparameters(model,opt,lr) 
    return model,optimizer,criterion

def dummy_model(architecture,n_classes,device):
    if architecture == 'VGG16':
        model = make_vgg('VGG16',n_classes)
    if architecture =='ResNet18':
        model = ResNet18(n_classes)
    elif architecture == 'CCTcifar':
        model = CCT(
            img_size = (32, 32),
            embedding_dim = 256,
            n_conv_layers = 2,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 6,
            num_heads = 4,
            mlp_ratio = 2.,
            num_classes = n_classes,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none'],
            n_input_channels=3,
        )
    elif architecture == 'ViTcifar':
        model = ViTcifar(num_classes = n_classes, dim = 512, depth = 6, heads = 6, mlp_dim = 1024)
    
    if model.device.type == 'cpu':
        model = model.to(device)
    return model

# Gets the loss of the model on datasets
def logits(model,train_loader,test_loader,device):
    model.eval()
    df_train_loss = pd.DataFrame()
    df_test_loss = pd.DataFrame()
    df_all_loss = pd.DataFrame()

    # Process training set
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(tqdm(train_loader)):
            if data.device.type == 'cpu':
                data = data.to(device)
            if target.device.type == 'cpu':
                target = target.to(device)
            logits_train = model(data)
            loss = F.cross_entropy(logits_train, target,reduction ='none')
            numpy_train_loss = loss.cpu().numpy()
            train_loss = pd.DataFrame(numpy_train_loss)
            df_train_loss = pd.concat([df_train_loss,train_loss],ignore_index=True)

    df_train_loss['label'] = 0    


    # Process test set
    with torch.no_grad():
        for data,target in test_loader:
            if data.device.type == 'cpu':
                data = data.to(device)
            if target.device.type == 'cpu':
                target = target.to(device)
            logits_test = model(data)
            loss = F.cross_entropy(logits_test, target,reduction ='none')
            numpy_test_loss = loss.cpu().numpy()
            test_loss = pd.DataFrame(numpy_test_loss)
            df_test_loss = pd.concat([df_test_loss,test_loss],ignore_index=True)
    df_test_loss['label'] = 1  

    df_all_loss = pd.concat([df_train_loss,df_test_loss],ignore_index=True)
    return df_all_loss

# Gets the loss of the model on the forget dataset
def logits_unlearn(model,forget_loader,device):
    model.eval()
    df_forget_loss = pd.DataFrame()

    # Process training set
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(tqdm(forget_loader)):
            if data.device.type == 'cpu':
                data = data.to(device)
            if target.device.type == 'cpu':
                target = target.to(device)
            logits = model(data)
            loss = F.cross_entropy(logits, target,reduction ='none')
            numpy_loss = loss.cpu().numpy()
            forget_loss = pd.DataFrame(numpy_loss)
            df_forget_loss = pd.concat([df_forget_loss,forget_loss],ignore_index=True)
    df_forget_loss['label'] = 1
    return df_forget_loss


# Provides accuracy score on a dataset for a model
def evaluate(model,dataloader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            if data.device.type == 'cpu':
                data = data.to(device)
            if target.device.type == 'cpu':
                target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Provides accuracy score on a dataset for a model
def evaluate_test(model,test_loader,criterion,n_classes,device):
    metric = MulticlassCalibrationError(n_classes, n_bins=15, norm='l1')
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    ece = 0
    with torch.no_grad():
        for data, target in test_loader:
            if data.device.type == 'cpu':
                data = data.to(device)
            if target.device.type == 'cpu':
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








     