import torch
import utils
from tqdm import tqdm
from copy import deepcopy
from torchmetrics.classification import MulticlassCalibrationError
import time 

# Class to do gradient ascent and amnesaic impair followed by fine tuning (which is used as the repair step for all unlearning methods)

class Unlearner():
    def __init__(self,model,remain_loader, remain_eval_loader, forget_loader,forget_eval_loader,test_loader, optimizer, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed):
        self.model = model
        self.remain_loader = remain_loader
        self.remain_eval_loader = remain_eval_loader
        self.forget_loader = forget_loader
        self.forget_eval_loader = forget_eval_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.n_epoch_impair = n_epoch_impair
        self.n_epoch_repair = n_epoch_repair
        self.n_classes = n_classes
        self.seed = seed

    def evaluate(self,dataloader):
        self.model.eval()
        model_loss = 0.0
        correct = 0
        total = 0
        ece = 0
        ece = MulticlassCalibrationError(self.n_classes, n_bins=15, norm='l1')
        for data, target in dataloader:
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                ece.update(torch.softmax(output, dim=1),target)
                model_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        ece = ece.compute().item()
        model_loss /= len(dataloader)
        accuracy = 100 * correct / total
        return accuracy,model_loss, ece
    
    def gradient_ascent(self):
        impair_time = 0
        epoch_time = 0
        for epoch in tqdm(range(self.n_epoch_impair)):
            start_time = time.time()
            train_loss = 0.0
            self.model.train()

            for batch_idx,(data,target) in enumerate(self.forget_loader):
                if data.device.type == 'cpu':
                    data = data.to(self.device)
                if target.device.type == 'cpu':
                    target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = -self.criterion(output,target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            end_time = time.time()
            epoch_time =  end_time - start_time
            impair_time += round(epoch_time,3)
            foget_accuracy,forget_loss,forget_ece = self.evaluate(self.forget_eval_loader)
            remain_accuracy,remain_loss,remain_ece  = self.evaluate(self.remain_eval_loader)
            test_accuracy,test_loss,test_ece  = self.evaluate(self.test_loader)
            print(f"Epoch: {epoch}/{self.n_epoch_impair}\tForget accuracy: {foget_accuracy:.2f}%\tForget loss: {forget_loss:.6f}")
            print(f'Remain accuracy: {remain_accuracy:.2f}%\tRemain loss: {remain_loss:.6f}\tRemain ECE: {remain_ece:.6f}')
            print(f'Test accuracy: {test_accuracy:.2f}%\tTest loss: {test_loss:.6f}\tTest ECE: {test_ece:.6f}')
        return self.model,impair_time

    def fine_tune(self):
        
        utils.set_seed(self.seed)
        train_ece = 0 
        test_ece = 0
        fine_tune_time = 0
        epoch_time = 0

        for epoch in tqdm(range(0, self.n_epoch_repair)):
            start_time = time.time()
            
            self.model.train()

            for batch_idx, (data, target) in enumerate(self.remain_loader):
                if data.device.type == 'cpu':
                    data = data.to(self.device)
                if target.device.type == 'cpu':
                    target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            end_time = time.time()
            epoch_time = end_time - start_time
            fine_tune_time += round(epoch_time,3)

            train_accuracy,train_loss,train_ece = self.evaluate(self.remain_eval_loader)
            test_accuracy,test_loss, test_ece= self.evaluate(self.test_loader)

            print(f"Epoch: {epoch}/{self.n_epoch_repair}\tTrain accuracy: {train_accuracy:.2f}%\tTrain loss: {train_loss:.6f}\tTrain ECE {train_ece:.2f}")
            print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%\tTest ECE {test_ece:.2f}"')

        return self.model,train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,self.n_epoch_repair,fine_tune_time
    
    def amnesiac(self):
        
        utils.set_seed(self.seed)
        train_ece = 0 
        test_ece = 0
        impair_time = 0
        epoch_time = 0
        
        for epoch in tqdm(range(0, self.n_epoch_impair)):
            start_time = time.time()
            self.model.train()
            self.model.train()

            for batch_idx, (data, target) in enumerate(self.forget_loader):
                if data.device.type == 'cpu':
                    data = data.to(self.device)
                if target.device.type == 'cpu':
                    target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            end_time = time.time()
            epoch_time = end_time - start_time
            impair_time += round(epoch_time,3)

            train_accuracy,train_loss,train_ece = self.evaluate(self.forget_eval_loader)
            test_accuracy,test_loss, test_ece= self.evaluate(self.test_loader)
                
            print(f"Epoch: {epoch}/{self.n_epoch_repair}\t Forget random labels accuracy: {train_accuracy:.2f}%\Forget random labels loss: {train_loss:.6f}\Forget random labels ECE {train_ece:.2f}")
            print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%\tTest ECE {test_ece:.2f}"')

        return self.model,impair_time