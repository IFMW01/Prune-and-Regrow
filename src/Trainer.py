import torch
import utils
from tqdm import tqdm
from copy import deepcopy
import time
from torchmetrics.classification import MulticlassCalibrationError

class Trainer():
    def __init__(self,model, train_loader, train_eval_loader, test_loader, optimizer, criterion, device, n_epoch,n_classes,seed):
        self.model = model
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.n_epoch = n_epoch
        self.n_classes = n_classes
        self.seed = seed
        self.metric = MulticlassCalibrationError(self.n_classes, n_bins=15, norm='l1')

    def evaluate(self,dataloader):
        self.model.eval()
        model_loss = 0.0
        correct = 0
        total = 0
        ece = 0
        for data, target in dataloader:
            with torch.no_grad():
                # data = data.to(self.device)
                # target = target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                ece += self.metric(output,target).item()
                model_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        ece /= len(dataloader)
        model_loss /= len(dataloader)
        accuracy = 100 * correct / total
        return accuracy,model_loss, ece

    def train(self):
        
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
        training_time = 0
        best_time = 0
        
        losses = []
        accuracies = []

        for epoch in tqdm(range(0, self.n_epoch)):
            epoch_time = 0
            start_time = time.time()
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                # data = data.to(self.device)
                # target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            training_time +=  round(epoch_time, 3)

            train_accuracy,train_loss,train_ece = self.evaluate(self.train_eval_loader)
            accuracies.append(train_accuracy)
            test_accuracy,test_loss, test_ece= self.evaluate(self.test_loader)
            if test_accuracy > best_test_accuracy:
                best_time = training_time
                best_test_accuracy = test_accuracy
                best_test_loss = test_loss
                best_model = deepcopy(self.model)
                best_model_epoch = epoch
                best_train_accuracy = train_accuracy
                best_train_loss = train_loss
                best_train_ece = train_ece
                best_test_ece = test_ece
                
            losses.append(train_loss)
            print(f"Epoch: {epoch}/{self.n_epoch}\tTrain accuracy: {train_accuracy:.2f}%\tTrain loss: {train_loss:.6f}\tTrain ECE {train_ece:.2f}")
            print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%\tTest ECE {test_ece:.2f}"')


        print(f"Best model achieved at epoch: {best_model_epoch}\t Train accuracy: {best_train_accuracy:.2f}\t Test accuracy: {best_test_accuracy:.2f}")

        return best_model,best_train_accuracy,best_train_loss,best_train_ece,best_test_accuracy,best_test_loss,best_test_ece,best_model_epoch,best_time
    