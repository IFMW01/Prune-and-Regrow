import vgg
import torch
import torch.optim as optim
import training as tr
import load_datasets
import random
import main as m
import utils
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
    naive_model,optimizer,scheduler,criterion = utils.initialise_model(architecture,in_channels,num_classes,device)
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
  
def fine_tuning(model, remain_loader,forget_loader,test_loader,optimizer,criterion, device, n_epoch):
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

    return model

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

    optimizer,scheduler,criterion = utils.set_hyperparameters(model,lr=0.05)
    print("\nFINE TUNING")
    model = fine_tuning(model, remain_loader,forget_loader,test_loader,optimizer,criterion, device, 4)
    return model

# FINE TUNE UNLEARNING

def fine_tuning_unlearning(path,architecture,in_channels,num_classes,device,remain_loader,forget_loader,test_loader,n_epoch):
   model,optimizer,criterion = load_model(path,architecture,in_channels,num_classes,device)
   model = fine_tuning(model, remain_loader,forget_loader,test_loader,optimizer,criterion, device, n_epoch)
   return model


# KD UNLEARNING

def train_knowledge_distillation(optimizer,criterion,teacher, student, train_loader, epochs, T, soft_target_loss_weight,ce_loss_weight,device):
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    teacher.to(device)
    student.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)
            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = criterion(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    return student

   
   