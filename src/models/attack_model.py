# https://github.com/teobaluta/etio/blob/main/attack/attack_models.py

#  shokri_shadow_model_attack as attack models trained using the data we obtain from the shadow models

import json
import sys
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxModel(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, n_out)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        y = x
        return y

def softmax_net(n_inputs):
    # width=int(config["width"])
    net = SoftmaxModel(n_in=n_inputs, n_out = 2)
    return net

class SimpleNet(nn.Module):

    def __init__(self, n_inputs, hidden_layers, out_dim=2):
        super(SimpleNet, self).__init__()
        self.layers = [nn.Linear(n_inputs, hidden_layers), nn.ReLU(inplace=True)]
        for i in range(1,len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*self.layers,
            nn.Linear(hidden_layers[-1], out_dim)
        )


    def forward(self, x):
        x = self.net(x)
        return x

def simplenet(n_inputs,hidden_layers):
    # width=int(config["width"])
    net = SimpleNet(hidden_layers, n_inputs)
    return net
