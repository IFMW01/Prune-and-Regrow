import vgg
import torch
import torch.optim as optim
import training as tr
import load_datasets
import random
import main as m
import utils as ut
from vgg import VGGish, VGG9
from tqdm import tqdm

def load_model(path,architecture,in_channels,num_classes,device):
  if architecture == 'VGGish':
    model = VGGish(in_channels,num_classes)
  elif architecture == 'VGG9':
    model = VGG9(in_channels,num_classes)
  model.load_state_dict(torch.load(path))
  model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.005,momentum=0.9)
  criterion = torch.nn.CrossEntropyLoss()
  return model,optimizer,criterion

def create_forget_set(forget_instances_num,train_set,seed):
  tr.set_seed(seed)
  forget_set = []
  remain_set = train_set
  for i in range(forget_instances_num):
    index = random.randint(0, (len(remain_set)-1))
    forget_set.append(remain_set[i])
    remain_set.pop(index)
  return forget_set, remain_set

def evaluate_forget_set(model,forget_loader,remain_loader,test_loader,device):
    forget_set_acc = tr.evaluate(model, forget_loader, device)
    print(f"Staring forget set Accuracy: {forget_set_acc:.2f}%")
    remain_set_acc = tr.evaluate(model, remain_loader, device)
    print(f"Staring remain set Accuracy: {remain_set_acc:.2f}%")
    test_set_acc = tr.evaluate(model, test_loader, device)
    print(f"Staring test set Accuracy: {test_set_acc:.2f}")

def naive_unlearning(architecture,in_channels,num_classes,device,remain_loader,forget_loader,test_loader,optimizer,criterion, n_epoch,seed):
    naive_model,optimizer,scheduler,criterion = ut.initialise_model(architecture,in_channels,num_classes,device)
    naive_model.to(device)
    losses = []
    accuracies = []
    evaluate_forget_set(naive_model,forget_loader,remain_loader,test_loader,device)

    for epoch in tqdm(range(1, n_epoch + 1)):
        naive_model.train()
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(remain_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = naive_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(remain_loader)
        accuracy = tr.evaluate(naive_model, remain_loader, device)
        accuracies.append(accuracy)

        losses.append(epoch_loss)

        test_loss, test_accuracy = tr.evaluate_test(naive_model, test_loader, criterion, device)
        forget_loss, forget_accuracy = tr.evaluate_test(naive_model, forget_loader, criterion, device)
        print(f"Epoch: {epoch}/{n_epoch}\tRemain Loss: {epoch_loss:.6f}\tRemain Accuracy: {accuracy:.2f}%")
        print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%')
        print(f'Forget Loss: {forget_loss:.6f}, Forget Accuracy: {forget_accuracy:.2f}%')
  
def fine_tuning(model, remain_loader,forget_loader,test_loader,optimizer,criterion, device, n_epoch, log_interval,seed):
    losses = []
    accuracies = []
    model.to(device)
    evaluate_forget_set(model,forget_loader,remain_loader,test_loader,device)
    for epoch in tqdm(range(1, n_epoch + 1)):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(remain_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(remain_loader)
        accuracy = tr.evaluate(model, remain_loader, device)
        accuracies.append(accuracy)

        losses.append(epoch_loss)

        test_loss, test_accuracy = tr.evaluate_test(model, test_loader, criterion, device)
        forget_loss, forget_accuracy = tr.evaluate_test(model, forget_loader, criterion, device)
        print(f"Epoch: {epoch}/{n_epoch}\tRemain Loss: {epoch_loss:.6f}\tRemain Accuracy: {accuracy:.2f}%")
        print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%')
        print(f'Forget Loss: {forget_loss:.6f}, Forget Accuracy: {forget_accuracy:.2f}%')

    # return model

def gradient_ascent(model,remain_loader,test_loader,forget_loader, optimizer, criterion, device, n_epoch, log_interval, seed):
    tr.set_seed(seed)
    model.to(device)
    losses = []
    accuracies = []
    evaluate_forget_set(model,forget_loader,remain_loader,test_loader,device)
    for epoch in tqdm(range(1, n_epoch + 1)):
        epoch_loss = 0.0
        model.train()

        for batch_idx, (data, target) in enumerate(forget_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = -criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(forget_loader)
        accuracy = tr.evaluate(model, forget_loader, device)
        accuracies.append(accuracy)

        losses.append(epoch_loss)

        remain_loss, remain_accuracy = tr.evaluate_test(model, remain_loader, criterion, device)
        test_loss, test_accuracy = tr.evaluate_test(model, test_loader, criterion, device)
        print(f"Epoch: {epoch}/{n_epoch}\tForget Loss: {epoch_loss:.6f}\tForget Accuracy: {accuracy:.2f}%")
        print(f'Remain Loss: {remain_loss:.6f}, Remain Accuracy: {remain_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%')
    return model