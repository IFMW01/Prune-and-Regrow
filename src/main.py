import torch.optim as optim
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_hyperparameters(model):
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device