import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torchaudio
import os
import librosa
import numpy as np
import torchvision
import random
import torchvision.datasets as cifar_datasets
import torchvision.transforms as transforms
import utils

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_datasets(dataset_pointer :str,unlearnng:bool):
    if dataset_pointer =="CIFAR10":
        base_transformations = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        train_set =  cifar_datasets.CIFAR10(
            root ='./',
            train=True,
            download=True,
            transform= base_transformations,
        )

        test_set = cifar_datasets.CIFAR10(
            root ='./',
            train=False,
            download=True,
            transform=base_transformations,
        )
    elif dataset_pointer =="CIFAR100":
        base_transformations = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        train_set =  cifar_datasets.CIFAR100(
            root ='./',
            train=True,
            download=True,
            transform= base_transformations,
        )

        test_set = cifar_datasets.CIFAR100(
            root ='./',
            train=False,
            download=True,
            transform=base_transformations,
        )

    else:
        raise Exception("Enter valid dataset pointer: e.g. CIFAR10, CIFAR100 or TinyImageNet")
    
    if unlearnng:
        return train_set,test_set
    device  = utils.get_device()
        
    generator = torch.Generator()
    generator.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=256,shuffle=True,worker_init_fn=seed_worker,
        generator=generator)
    train_eval_loader = DataLoader(train_set, batch_size=256,shuffle=False,worker_init_fn=seed_worker,
        generator=generator)
    test_loader = DataLoader(test_set, batch_size=256,shuffle=False,worker_init_fn=seed_worker,
        generator=generator)
        
    return train_loader,train_eval_loader,test_loader


class DatasetProcessor_randl_cifar(Dataset):
  def __init__(self, dataset,device,num_classes):
    self.dataset = dataset
    self.data = []
    self.labels = []
    for inx, (data, label) in enumerate(dataset):
        self.data.append(data.to(device))
        new_label = label
        while new_label == label:
                new_label = random.randint(0, (num_classes-1))
        new_label = torch.tensor(new_label).to(device)
        self.labels.append(new_label)

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx] 





