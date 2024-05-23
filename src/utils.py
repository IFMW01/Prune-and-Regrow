import torch.optim as optim
import torch
import torch.nn as nn
import os
from models.vgg import VGGishMel,VGGishSpec,VGG9,VGGishMelr,VGGishSpecr
from models.transformer import ViTmel, ViTspec
from models.compact_ViT import CCT
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_hyperparameters(model,architecture,lr):
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
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
    elif architecture == 'ViTmel':
        model = ViTmel(
        num_classes = n_classes,
        dim = 512,
        depth = 6,
        heads = 6,
        mlp_dim = 1024
        )
    elif architecture == 'ViTspec':
        model = ViTspec(
        num_classes = n_classes,
        dim = 512,
        depth = 6,
        heads = 6,
        mlp_dim = 1024
        )
    elif architecture == 'CTCmel':
        model = CCT(
            img_size = (32, 63),
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
            n_input_channels=1,
        )
    elif architecture == 'CTCspec':
        model= CCT(
            img_size = (257, 63),
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
            n_input_channels=1,
        )
    elif architecture == 'VGG9':
        model = VGG9()

    model = model.to(device)
    optimizer,criterion = set_hyperparameters(model,architecture,lr) 
    return model,optimizer,criterion

def logits(model,train_loader,test_loader,device):
    model.eval()
    # df_train_logits = pd.DataFrame()
    # df_test_logits = pd.DataFrame()
    # df_all_logits = pd.DataFrame()
    df_train_loss = pd.DataFrame()
    df_test_loss = pd.DataFrame()
    df_all_loss = pd.DataFrame()

    # Process training set
    for batch_idx,(data,target) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            logits_train = model(data)
            # logits_train_softmax = F.softmax(logits_train,dim=1)
            loss = F.cross_entropy(logits_train, target,reduction ='none')
            numpy_train_loss = loss.cpu().numpy()
            train_loss = pd.DataFrame(numpy_train_loss)
            df_train_loss = pd.concat([df_train_loss,train_loss],ignore_index=True)

    df_train_loss['label'] = 0    


    # Process test set
    for data,target in test_loader:
        with torch.no_grad():
            logits_test = model(data)
            loss = F.cross_entropy(logits_test, target,reduction ='none')
            numpy_test_loss = loss.cpu().numpy()
            test_loss = pd.DataFrame(numpy_test_loss)
            df_test_loss = pd.concat([df_test_loss,test_loss],ignore_index=True)
    df_test_loss['label'] = 1  

    df_all_loss = pd.concat([df_train_loss,df_test_loss],ignore_index=True)
    return df_all_loss

def logits_unlearn(model,forget_loader,device):
    model.eval()
    df_forget_loss = pd.DataFrame()

    # Process training set
    for batch_idx,(data,target) in enumerate(tqdm(forget_loader)):
        with torch.no_grad():
            logits = model(data)
            loss = F.cross_entropy(logits, target,reduction ='none')
            numpy_loss = loss.cpu().numpy()
            forget_loss = pd.DataFrame(numpy_loss)
            df_forget_loss = pd.concat([df_forget_loss,forget_loss],ignore_index=True)
    df_forget_loss['label'] = 1
    return df_forget_loss

def evaluate(model,dataloader,device):
    model.eval()
    correct = 0
    total = 0
    for data, target in dataloader:
        with torch.no_grad():
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








     