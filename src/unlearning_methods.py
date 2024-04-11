import torch
import torch.nn as nn
import torch.optim as optim
import training as tr
import random
import utils
import numpy as np
from vgg import VGGish,VGG9
from tqdm import tqdm
from copy import deepcopy
import torchmetrics.classification 
from torchmetrics.classification import MulticlassCalibrationError


def load_model(path,device):
  model = torch.load(path)
  
  model.to(device)
  optimizer = optim.SGD(model.parameters(),lr=0.005,momentum=0.9)
  criterion = torch.nn.CrossEntropyLoss()
  return model,optimizer,criterion

def create_forget_remain_set(forget_instances_num,train_set,seed=42):
  utils.set_seed(seed)
  forget_set = []
  remain_set = train_set
  for i in range(forget_instances_num):
    index = random.randint(0,(len(remain_set)-1))
    forget_set.append(remain_set[i])
    remain_set.pop(index)
  return forget_set,remain_set

def evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device):
    forget_set_acc = tr.evaluate(model,forget_loader,device)
    print(f"Forget set Accuracy: {forget_set_acc:.2f}%")
    remain_set_acc = tr.evaluate(model,remain_loader,device)
    print(f"Remain set Accuracy: {remain_set_acc:.2f}%")
    test_set_acc = tr.evaluate(model,test_loader,device)
    print(f"Test set Accuracy: {test_set_acc:.2f}")

# NAIVE  UNLEARNING
def naive_unlearning(architecture,in_channels,n_classes,device,remain_loader,forget_loader,test_loader,n_epochs,results_dict,seed):
    print("\nNaive Unlearning:")
    print("\n")
    utils.set_seed(seed)

    naive_model,optimizer,scheduler,criterion = utils.initialise_model(architecture,in_channels,n_classes,device)
    evaluate_forget_remain_test(naive_model,forget_loader,remain_loader,test_loader,device)

    naive_model,train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece = tr.train(naive_model,remain_loader,test_loader,optimizer,criterion,device,n_epochs,seed)

    evaluate_forget_remain_test(naive_model,forget_loader,remain_loader,test_loader,device)
    forget_accuracy,forget_loss,forget_ece = tr.evaluate_test(naive_model,forget_loader,criterion,n_classes,device)

    results_dict['Naive Unlearning'] = [train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]

    return naive_model,results_dict
  
def fine_tuning(model,remain_loader,forget_loader,test_loader,optimizer,criterion,n_epochs,n_classes,device):
    metric = MulticlassCalibrationError(n_classes=n_classes,n_bins=15,norm='l1')
    print("\nFine Tuning:")
    losses = []
    accuracies = []
    train_ece = 0
    test_ece = 0
    forget_ece = 0
    model.to(device)
    evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device)
    for epoch in tqdm(range(1,n_epochs + 1)):
        model.train()
        train_loss = 0.0

        for batch_idx,(data,target) in enumerate(remain_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_ece += metric(output,target)
            train_loss += loss.item()
        train_ece /= len(remain_loader) 
        train_loss /= len(remain_loader)
        train_accuracy = tr.evaluate(model,remain_loader,device)
        accuracies.append(train_accuracy)

        losses.append(train_loss)

        test_accuracy,test_loss,test_ece = tr.evaluate_test(model,test_loader,criterion,n_classes,device)
        forget_accuracy,forget_loss,forget_ece  = tr.evaluate_test(model,forget_loader,criterion,n_classes,device)
        print(f"Epoch: {epoch}/{n_epochs}\tRemain Loss: {train_loss:.6f}\tRemain Accuracy: {train_accuracy:.2f}%\tRemain ECE: {train_ece:.2f}")
        print(f"Test Loss:{test_loss:.6f}\tTest Accuracy: {test_accuracy:.2f}%\tTest ECE: {test_ece:.2f}")
        print(f"Forget Loss: {forget_loss:.6f}\tForget Accuracy: {forget_accuracy:.2f}%\tForget ECE {forget_ece:.2F}")
    
    return model,train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece

# GRADIENT ASCENT UNLEARNING

def gradient_ascent(path,remain_loader,test_loader,forget_loader,device,n_epoch_impair,n_epoch_repair,results_dict,n_classes,seed):
    print("\nGradient Ascent Unlearning:")
    print("\n")
    utils.set_seed(seed)
    model,optimizer,criterion = load_model(path,device)
    model.to(device)
    losses = []
    accuracies = []
    evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device)
    for epoch in tqdm(range(n_epoch_impair)):
        train_loss = 0.0
        model.train()

        for batch_idx,(data,target) in enumerate(forget_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = -criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(forget_loader)
        train_accuracy = tr.evaluate(model,forget_loader,device)
        accuracies.append(train_accuracy)

        losses.append(train_loss)

        remain_accuracy,remain_loss,remain_ece  = tr.evaluate_test(model,remain_loader,criterion,n_classes,device)
        test_accuracy,test_loss,test_ece  = tr.evaluate_test(model,test_loader,criterion,n_classes,device)
        print(f"Epoch: {epoch}/{n_epoch_impair}\tForget Loss: {train_loss:.6f}\tForget Accuracy: {train_accuracy:.2f}%")
        print(f'Remain Loss: {remain_loss:.6f},Remain Accuracy: {remain_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.6f},Test Accuracy: {test_accuracy:.2f}%')

    optimizer,scheduler,criterion = utils.set_hyperparameters(model,lr=0.05)
    model,train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece = fine_tuning(model,remain_loader,forget_loader,test_loader,optimizer,criterion,n_epoch_repair,n_classes,device)
    results_dict['Gradient Ascent Unlearning'] = [train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return model,results_dict

# FINE TUNE UNLEARNING

def fine_tuning_unlearning(path,device,remain_loader,forget_loader,test_loader,n_epochs,results_dict,n_classes,seed):
   print("\nFine Tuning Unlearning:")
   utils.set_seed(seed)
   model,optimizer,criterion = load_model(path,device)
   model,train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece = fine_tuning(model,remain_loader,forget_loader,test_loader,optimizer,criterion,n_epochs,n_classes,device)
   results_dict['Fine Tune Unlearning'] = [train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
   return model,results_dict


# STOCHASTIC TEACHER UNLEARNING

def train_knowledge_distillation(optimizer,criterion,teacher,student,train_loader,epochs,T,soft_target_loss_weight,ce_loss_weight,device):
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    teacher.to(device)
    student.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs,labels in train_loader:
            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)
            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T,dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T,dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = criterion(student_logits,labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs},Loss: {running_loss / len(train_loader)}")
    return student

def stochastic_teacher_unlearning(path,forget_loader,remain_loader,test_loader,device,in_channels,n_classes,architecture,results_dict,seed):
  print("\nStochastic Teacher Unlearning:")
  print("\n")
  utils.set_seed(seed)
  model,optimizer,criterion,= load_model(path,device)
  model.to(device)
  optimizer_bt = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
  optimizer_gt = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)


  if architecture =='VGGish':
    stochastic_teacher = VGGish(in_channels,n_classes)

  evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device)
  orignial_model = deepcopy(model)
  erased_model = train_knowledge_distillation(optimizer_bt,criterion,teacher=stochastic_teacher,student=model,train_loader=forget_loader,epochs=1,T=1,soft_target_loss_weight=0.5,ce_loss_weight=0.5,device=device)
  evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device)

  retrained_model = train_knowledge_distillation(optimizer_gt,criterion,teacher=orignial_model,student=erased_model,train_loader=remain_loader,epochs=1,T=1,soft_target_loss_weight=0,ce_loss_weight=1,device=device)

  erased_forget_acc = tr.evaluate(erased_model,forget_loader,device)
  print(f"Erased model forget set ACC: {erased_forget_acc}")

  forget_accuracy,forget_loss,forget_ece  = retrained_forget_acc = tr.evaluate_test(retrained_model,forget_loader,n_classes,device)
  print(f"Retrained model forget set ACC: {retrained_forget_acc}")

  train_accuracy,train_loss,train_ece = tr.evaluate_test(retrained_model,remain_loader,n_classes,device)
  print(f"Retrained model remain set ACC: {train_accuracy}")

  test_accuracy,test_loss,test_ece = tr.evaluate_test(retrained_model,test_loader,n_classes,device)
  print(f"Retrained model test set ACC: {test_accuracy}")
  results_dict['Stochastic Teacher Unlearning'] = [train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
  return model,results_dict

  # ONE-SHOT MAGNITUTE UNLEARNING
  
def global_unstructured_pruning(model,pruning_ratio):
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.data.view(-1))
    all_weights = torch.cat(all_weights)
    threshold = np.percentile(all_weights.cpu().numpy(),pruning_ratio)

    for param in model.parameters():
        param.data[param.data.abs() < threshold] = 0
    return model

def omp_unlearning(path,device,forget_loader,remain_loader,test_loader,pruning_ratio,n_epochs,results_dict,n_classes,seed):
    print("\nOMP Unlearning:")
    print("\n")
    utils.set_seed(seed)
    model,optimizer,criterion,= load_model(path,device)
    model = global_unstructured_pruning(model,pruning_ratio)
    print("Pruning Complete:")
    evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device)
    model,train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece  = fine_tuning(model,remain_loader,forget_loader,test_loader,optimizer,criterion,n_epochs,n_classes,device)
    results_dict["OMP Unlearning"] = [train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return model,results_dict

  # CONSINE OMP PRUNE UNLEARNING
