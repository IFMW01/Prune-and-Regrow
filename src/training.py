import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from copy import deepcopy
import torchmetrics.classification 
from torchmetrics.classification import MulticlassCalibrationError


def evaluate(model, dataloader, device):
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

def evaluate_test(model, test_loader, criterion,n_classes,device):
    
    metric = MulticlassCalibrationError(num_classes=n_classes, n_bins=15, norm='l1')
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
            ece += metric(output,target)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    ece /= len(test_loader)
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    return test_accuracy,test_loss, ece

def train(model, train_loader, test_loader, optimizer, criterion, device, n_epoch,n_classes,seed):
    metric = MulticlassCalibrationError(num_classes=n_classes, n_bins=15, norm='l1')
    utils.set_seed(seed)
    train_ece = 0 
    test_ece = 0
    best_model = None
    best_model_epoch = 0
    best_test_accuracy = 0
    best_train_accuracy = 0
    best_train_loss = 0
    best_train_ece = 0
    best_test_ece = 0


    losses = []
    accuracies = []

    for epoch in tqdm(range(1, n_epoch + 1)):
      model.train()
      epoch_loss = 0.0

      for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_ece += metric(output, target)
        epoch_loss += loss.item()

      accuracy = evaluate(model, train_loader, device)
      accuracies.append(accuracy)
      train_ece /= len(train_loader)
      test_accuracy,test_loss, test_ece= evaluate_test(model, test_loader, criterion,n_classes,device)
      
      if test_accuracy > best_test_accuracy:
          best_test_accuracy = test_accuracy
          best_test_loss = test_loss
          best_model = deepcopy(model)
          best_model_epoch = epoch
          best_train_accuracy = accuracy
          best_train_loss = loss.item()
          best_train_ece = train_ece
          best_test_ece = test_ece

      epoch_loss /= len(train_loader)
      losses.append(epoch_loss)
      print(f"Epoch: {epoch}/{n_epoch}\tTrain loss: {epoch_loss:.6f}\tTrain accuracy: {accuracy:.2f}%")
      print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%')
      

    print(f"Best model achieved at epoch: {best_model_epoch}\t Train accuracy: {best_train_accuracy:.2f}\t Test accuracy: {best_test_accuracy:.2f}")
    # plt.plot(losses)
    # plt.title("Training Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

    return best_model,best_train_accuracy,best_train_loss,best_train_ece,best_test_accuracy,best_test_loss,best_test_ece
