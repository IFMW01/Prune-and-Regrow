import torch.optim as optim
import torch
import torch.nn as nn
import os
import vgg 
from vgg import VGGish, VGG9
import numpy as np
import random

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
    optimizer = optim.SGD(model.parameters(), lr,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    return optimizer, scheduler,criterion

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
    optimizer, scheduler,criterion = set_hyperparameters(model,lr) 
    return model,optimizer, scheduler,criterion