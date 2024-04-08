import vgg
import torch
import torch.optim as optim
import training as tr
import load_datasets
import random
import main as m
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

def create_forget_set(forget_instances,train_set,seed):
  tr.set_seed(seed)
  forget_set = []
  remain_set = train_set
  for i in range(forget_instances):
    index = random.randint(0, (len(remain_set)-1))
    forget_set.append(remain_set[i])
    remain_set.pop(index)
  return forget_set, remain_set

def evaluate_forget_set(model,forget_loader,remain_loader,test_loader,device):
    forget_set_acc = tr.evaluate(model, forget_loader, device)
    print(f"Staring forget set Accuracy: {forget_set_acc}")
    remain_set_acc = tr.evaluate(model, remain_loader, device)
    print(f"Staring remain set Accuracy: {remain_set_acc}")
    test_set_acc = tr.evaluate(model, test_loader, device)
    print(f"Staring test set Accuracy: {test_set_acc}")

def naive(architecture,in_channels,num_classes,device,remain_loader,forget_loader,test_loader,optimizer,criterion, device, n_epoch,seed):
    naive_model = m.initialise_model(architecture,in_channels,num_classes,device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    tr.set_seed(seed)
    losses = []
    accuracies = []
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
        accuracy = evaluate(model, remain_loader, device)
        accuracies.append(accuracy)

        losses.append(epoch_loss)

        test_loss, test_accuracy = evaluate_test(model, test_loader, criterion, device)
        forget_loss, forget_accuracy = evaluate_test(model, forget_loader, criterion, device)
        print(f"Epoch: {epoch}/{n_epoch}\tRemain Loss: {epoch_loss:.6f}\tRemain Accuracy: {accuracy:.2f}%")
        print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%')
        print(f'Forget Loss: {forget_loss:.6f}, Forget Accuracy: {forget_accuracy:.2f}%')