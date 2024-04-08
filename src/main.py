import torch.optim as optim
import torch
import torch.nn as nn
import json
import os
import training as tr
import load_datasets as ld
import membership_inference as mi
from vgg import VGGish,VGG9
# from transformer import SimpleViT

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_hyperparameters(model):
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    return optimizer, scheduler,criterion

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialise_model(architecture,n_inputs,n_classes):
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
    return model

def create_base_model(model,dataset_pointer,pipeline,save_path,device, n_epochs, seed):
        train_loader,valid_loader,test_loader,labels = ld.load_datasets(dataset_pointer,pipeline)
        optimizer, scheduler,criterion = set_hyperparameters(model)
        best_model,accuracies = tr.train(model, train_loader,valid_loader, test_loader, optimizer, criterion, device, n_epochs, seed)
        torch.save(best_model, f"{save_path}.pth")
        df_softmax_outputs = mi.mai_logits(best_model, train_loader, test_loader,device)
        df_softmax_outputs.to_csv(f'{save_path}_softmax_outputs.csv',index = False)


def main(config):
    dataset_pointer = config.get("dataset_pointer", None)
    pipeline = config.get("pipeline", None)
    architecture = config.get("architecture", None)
    n_epochs = config.get("n_epochs", None)
    training = config.get("training", None)
    seeds = config.get("seeds", None)
    n_classes = config.get("n_classes", None)
    n_inputs = config.get("n_inputs", None)

    print("Received arguments from config file:")
    print(f"Dataset pointer: {dataset_pointer}")
    print(f"Pipeline: {pipeline}")
    print(f"Architecture: {architecture}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Training: {training}")
    print(f"Seeds: {seeds}")

    device = get_device()
            
    if training == 'Base':
        save_dir = f"{training}_{dataset_pointer}"
        create_dir(save_dir)
        for i in range(len(seeds)):
            model = initialise_model(architecture,n_inputs,n_classes)
            seed = seeds[i]
            save_dir = os.path.join(f"{training}_{dataset_pointer}", f"{seed}")
            create_dir(save_dir)
            save_path = f"{save_dir}\{architecture}_{seed}"
            create_base_model(model,dataset_pointer,pipeline,save_path,device, n_epochs, seed)
    print("FIN")

if __name__ == "__main__":
    with open("./config.json", "r") as f:
        config = json.load(f)
    main(config)