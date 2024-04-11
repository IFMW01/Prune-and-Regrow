import torch
import utils
from tqdm import tqdm
from copy import deepcopy
import torchmetrics.classification 
from torchmetrics.classification import MulticlassCalibrationError

class Unlearner():
    def __init__(self,model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed):
        self.model = model
        self.remain_loader = remain_loader
        self.remain_eval_loader = remain_eval_loader
        self.forget_loader = forget_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.n_epoch_impair = n_epoch_impair
        self.n_epoch_repair = n_epoch_repair
        self.n_classes = n_classes
        self.seed = seed
        self.metric = MulticlassCalibrationError(self.n_classes, n_bins=15, norm='l1')

    def evaluate(self,dataloader):
        self.model.eval()
        model_loss = 0.0
        correct = 0
        total = 0
        ece = 0

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
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
    
    def gradient_ascent(self):

        for epoch in tqdm(range(self.n_epoch_impair)):
            train_loss = 0.0
            self.model.train()

            for batch_idx,(data,target) in enumerate(self.forget_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = -self.criterion(output,target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

 
            foget_accuracy,forget_loss,forget_ece = self.evaluate(self.forget_loader)
            remain_accuracy,remain_loss,remain_ece  = self.evaluate(self.remain_eval_loader)
            test_accuracy,test_loss,test_ece  = self.evaluate(self.test_loader)
            print(f"Epoch: {epoch}/{self.n_epoch_impair}\tForget accuracy: {foget_accuracy:.2f}%\tForget loss: {forget_loss:.6f}")
            print(f'Remain accuracy: {remain_accuracy:.2f}%\tRemain loss: {remain_loss:.6f}\tRemain ECE: {remain_ece:.6f}')
            print(f'Test accuracy: {test_accuracy:.2f}%\tTest loss: {test_loss:.6f}\tTest ECE: {test_ece:.6f}')
        return self.model

    def fine_tune(self):
        
        utils.set_seed(self.seed)
        train_ece = 0 
        test_ece = 0

        losses = []
        accuracies = []

        for epoch in tqdm(range(0, self.n_epoch_repair)):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.remain_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            train_accuracy,train_loss,train_ece = self.evaluate(self.remain_eval_loader)
            accuracies.append(train_accuracy)
            test_accuracy,test_loss, test_ece= self.evaluate(self.test_loader)
                
            losses.append(train_loss)
            print(f"Epoch: {epoch}/{self.n_epoch}\tTrain accuracy: {train_accuracy:.2f}%\tTrain loss: {train_loss:.6f}\tTrain ECE {train_ece:.2f}")
            print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%\tTest ECE {test_ece:.2f}"')

        return self.model,train_accuracy,train_loss,train_ece,test_accuracy,test_loss, test_ece