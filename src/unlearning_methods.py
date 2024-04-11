import torch
import torch.nn as nn
import torch.optim as optim
import random
import utils
import numpy as np
import Trainer
import Unlearner 
from copy import deepcopy
from Trainer import Trainer
from Unlearner import Unlearner



def create_forget_remain_set(forget_instances_num,train_set,seed=42):
    utils.set_seed(seed)
    forget_set = []
    remain_set = train_set
    for i in range(forget_instances_num):
        index = random.randint(0,(len(remain_set)-1))
        forget_set.append(remain_set[i])
        remain_set.pop(index)
    return remain_set,forget_set

def evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device):
    forget_set_acc = utils.evaluate(model,forget_loader,device)
    print(f"Forget set Accuracy: {forget_set_acc:.2f}%")
    remain_set_acc = utils.evaluate(model,remain_loader,device)
    print(f"Remain set Accuracy: {remain_set_acc:.2f}%")
    test_set_acc = utils.evaluate(model,test_loader,device)
    print(f"Test set Accuracy: {test_set_acc:.2f}")

def load_model(path,device):
    model = torch.load(path)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=0.005,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    return model,optimizer,criterion

# NAIVE  UNLEARNING
def naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs,results_dict,seed):
    print("\nNaive Unlearning:")
    print("\n")
    utils.set_seed(seed)
    naive_model,optimizer_nu,criterion = utils.initialise_model(architecture,n_inputs,n_classes,device)
    train_naive = Trainer(naive_model, remain_loader, remain_eval_loader, test_loader, optimizer_nu, criterion, device, n_epochs,n_classes,seed)
    naive_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece = train_naive.train()
    forget_accuracy,forget_loss,forget_ece = train_naive.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy}:.2f%\tForget loss:{forget_loss}:.2f\tForget ECE:{forget_ece}:.2f")
    results_dict['Naive Unlearning'] = [remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]

    return naive_model,results_dict


# GRADIENT ASCENT UNLEARNING

def gradient_ascent(path,remain_loader,remain_eval_loader,test_loader,forget_loader,device,n_epoch_impair,n_epoch_repair,results_dict,n_classes,seed):
    print("\nGradient Ascent Unlearning:")
    print("\n")
    utils.set_seed(seed)
    ga_model,optimizer_ga,criterion = load_model(path,device)
    evaluate_forget_remain_test(ga_model,forget_loader,remain_loader,test_loader,device)
    ga_train = Unlearner(ga_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_ga, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed)
    ga_model = ga_train.gradient_ascent()

    print("\nFine tuning gradient ascent model:")
    optimizer_ft,criterion = utils.set_hyperparameters(ga_model,lr=0.05)
    ga_fine_tune = Unlearner(ga_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_ft, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed)
    ga_model, remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss, test_ece= ga_fine_tune.fine_tune()
    forget_accuracy,forget_loss,forget_ece = ga_fine_tune.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy}:.2f%\tForget loss:{forget_loss}:.2f\tForget ECE:{forget_ece}:.2f")
    results_dict['Gradient Ascent Unlearning'] = [remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return ga_model,results_dict

# FINE TUNE UNLEARNING

def fine_tuning_unlearning(path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs,results_dict,n_classes,seed):
   print("\nFine Tuning Unlearning:")
   utils.set_seed(seed)
   ft_model,optimizer_ft,criterion = load_model(path,device)
   ft_train = Unlearner(ft_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_ft, criterion, device,0,n_epochs,n_classes,seed)
   ft_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece = ft_train.fine_tune()
   forget_accuracy,forget_loss,forget_ece= ft_train.evaluate(forget_loader)
   print(f"Forget accuracy:{forget_accuracy}:.2f%\tForget loss:{forget_loss}:.2f\tForget ECE:{forget_ece}:.2f")
   results_dict['Fine Tune Unlearning'] = [remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
   return ft_model,results_dict


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

def stochastic_teacher_unlearning(path,remain_loader,test_loader,forget_loader,device,n_inputs,n_classes,architecture,results_dict,n_impair_epochs,n_repair_epochs,seed):
  print("\nStochastic Teacher Unlearning:")
  print("\n")
  utils.set_seed(seed)
  st_model,optimizer,criterion,= load_model(path,device)
  optimizer_bt = optim.SGD(st_model.parameters(),lr=0.001,momentum=0.9)

  stochastic_teacher,stochastic_teacher_optimizer,stochastic_teacher_criterion= utils.initialise_model(architecture,n_inputs,n_classes,device,seed)
  evaluate_forget_remain_test(st_model,forget_loader,remain_loader,test_loader,device)

  orignial_model = deepcopy(st_model)
  erased_model = train_knowledge_distillation(optimizer_bt,criterion,teacher=stochastic_teacher,student=st_model,train_loader=forget_loader,epochs=n_impair_epochs,T=1,soft_target_loss_weight=0.5,ce_loss_weight=0.5,device=device)
  evaluate_forget_remain_test(st_model,forget_loader,remain_loader,test_loader,device)
  optimizer_gt = optim.SGD(erased_model.parameters(),lr=0.01,momentum=0.9)
  retrained_model = train_knowledge_distillation(optimizer_gt,criterion,teacher=orignial_model,student=erased_model,train_loader=remain_loader,epochs=n_repair_epochs,T=1,soft_target_loss_weight=0,ce_loss_weight=1,device=device)

  erased_forget_acc = utils.evaluate(erased_model,forget_loader,device)
  print(f"Erased model forget set ACC: {erased_forget_acc}")

  forget_accuracy,forget_loss,forget_ece  = retrained_forget_acc = utils.evaluate_test(retrained_model,forget_loader,n_classes,criterion,device)
  print(f"Forget accuracy:{forget_accuracy}:.2f%\tForget loss:{forget_loss}:.2f\tForget ECE:{forget_ece}:.2f")

  train_accuracy,train_loss,train_ece = utils.evaluate_test(retrained_model,remain_loader,n_classes,criterion,device)
  test_accuracy,test_loss,test_ece = utils.evaluate_test(retrained_model,test_loader,n_classes,criterion,device)

  results_dict['Stochastic Teacher Unlearning'] = [train_accuracy,train_loss,train_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
  return st_model,results_dict

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

def omp_unlearning(path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,pruning_ratio,n_epochs,results_dict,n_classes,seed):
    print("\nOMP Unlearning:")
    print("\n")
    utils.set_seed(seed)
    omp_model,optimizer_omp,criterion,= load_model(path,device)
    omp_model = global_unstructured_pruning(omp_model,pruning_ratio)
    print("Pruning Complete:")
    evaluate_forget_remain_test(omp_model,forget_loader,remain_loader,test_loader,device)
    print("\nFine tuning pruned model:")
    omp_train = Unlearner(omp_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_omp, criterion, device,0,n_epochs,n_classes,seed)
    omp_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss, test_ece= omp_train.fine_tune()
    forget_accuracy,forget_loss,forget_ece = omp_train.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy}:.2f%\tForget loss:{forget_loss}:.2f\tForget ECE:{forget_ece}:.2f")
    results_dict["OMP Unlearning"] = [remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return omp_model,results_dict

  # CONSINE OMP PRUNE UNLEARNING