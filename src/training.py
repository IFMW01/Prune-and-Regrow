import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from copy import deepcopy
import torchmetrics.classification 
from torchmetrics.classification import MulticlassCalibrationError

class Trainer():
    def __init__(self,model, train_loader, test_loader, optimizer, criterion, device, n_epoch,n_classes,seed):
        self.model = model,
        self.train_loader = train_loader,
        self.test_loader = test_loader,
        self.optimizer = optimizer,
        self.criterion = criterion
        self.device = device,
        self.n_epoch = n_epoch,
        self.n_classes = n_classes,
        self.seed = seed

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def evaluate_test(self):

        metric = MulticlassCalibrationError(num_classes=self.n_classes, n_bins=15, norm='l1')
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        ece = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                ece += metric(output,target).item()
                test_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        ece /= len(self.test_loader)
        test_loss /= len(self.test_loader)
        test_accuracy = 100 * correct / total
        return test_accuracy,test_loss, ece

    def train(self):
        metric = MulticlassCalibrationError(num_classes=self.n_classes, n_bins=15, norm='l1')
        utils.set_seed(self.seed)
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

        for epoch in tqdm(range(1, self.n_epoch + 1)):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_ece += metric(output, target)
                train_ece = train_ece.item()
                epoch_loss += loss.item()

        accuracy = self.evaluate()
        accuracies.append(accuracy)
        train_ece /= len(self.train_loader)
        test_accuracy,test_loss, test_ece= self.evaluate_test()
        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_loss = test_loss
            best_model = deepcopy(self.model)
            best_model_epoch = epoch
            best_train_accuracy = accuracy
            best_train_loss = loss.item()
            best_train_ece = train_ece
            best_test_ece = test_ece

        epoch_loss /= len(self.train_loader)
        losses.append(epoch_loss)
        print(f"Epoch: {epoch}/{self.n_epoch}\tTrain accuracy: {accuracy:.2f}%\tTrain loss: {epoch_loss:.6f}\tTrain ECE {train_ece:.2f}")
        print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%\tTest ECE {test_ece:.2f}"')
        

        print(f"Best model achieved at epoch: {best_model_epoch}\t Train accuracy: {best_train_accuracy:.2f}\t Test accuracy: {best_test_accuracy:.2f}")

        return best_model,best_train_accuracy,best_train_loss,best_train_ece,best_test_accuracy,best_test_loss,best_test_ece
