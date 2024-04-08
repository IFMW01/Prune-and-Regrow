import random
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def evaluate_test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy

def train(model, train_loader,valid_loader, test_loader, optimizer, criterion, device, n_epoch, seed):
    set_seed(seed)
    best_model = None
    best_accuracy = 0.0
    best_model_epoch = 0
    best_test_accuracy = 0,
    early_stop_accuracy = 0


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

        epoch_loss += loss.item()

      accuracy = evaluate(model, train_loader, device)
      accuracies.append(accuracy)
      valid_loss, valid_accuracy = evaluate_test(model, valid_loader, criterion, device)
      test_loss, test_accuracy = evaluate_test(model, test_loader, criterion, device)
      
      if valid_accuracy > best_accuracy:
          best_accuracy = valid_accuracy
          best_model = deepcopy(model)
          best_model_epoch = epoch
          best_test_loss, best_test_accuracy = evaluate_test(best_model, test_loader, criterion, device)
          early_stop_accuracy = accuracy

      epoch_loss /= len(train_loader)
      losses.append(epoch_loss)
      print(f"Epoch: {epoch}/{n_epoch}\tTrain loss: {epoch_loss:.6f}\tTrain accuracy: {accuracy:.2f}%")
      print(f'Validation loss: {valid_loss:.6f}, Validation accuracy: {valid_accuracy:.2f}%')
      print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%')
      

    print(f"Best model achieved at epoch: {best_model_epoch}\t Train accuracy: {early_stop_accuracy:.2f}\t Valid accuracy: {best_accuracy:.2f}\t Test accuracy: {best_test_accuracy:.2f}")
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    return best_model,accuracies
