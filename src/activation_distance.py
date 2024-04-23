import torch.nn as nn 
import torch
import torch.functional as F

def actv_dist(model1, model2, dataloader, device):
    sftmx = nn.Softmax(dim = 1)
    distances = []
    for batch in dataloader:
        x, _, _ = batch
        x = x.to(device)
        model1_out = model1(x)
        model2_out = model2(x)
        diff = torch.sqrt(torch.sum(torch.square(F.softmax(model1_out, dim = 1) - F.softmax(model2_out, dim = 1)), axis = 1))
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis = 0)
    return distances.mean()