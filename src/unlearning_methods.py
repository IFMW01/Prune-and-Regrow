import torch
import torch.nn as nn
import torch.optim as optim
import random
import utils
import numpy as np
import Trainer
import Unlearner 
import scipy.stats as stats
import torch.nn.utils.prune as prune
from copy import deepcopy
from Trainer import Trainer
from Unlearner import Unlearner
from torch.autograd import grad
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector as Params2Vec



def create_forget_remain_set(dataset_pointer,forget_instances_num,train_set,seed=42):
    utils.set_seed(seed)
    forget_set = []
    remain_set = train_set
    if dataset_pointer == 'CIFAR10':
        total_instances = len(remain_set)
        random_indices = np.random.choice(total_instances, forget_instances_num, replace=False)
        forget_set = [remain_set[i] for i in random_indices]
        remain_set = [instance for i, instance in enumerate(remain_set) if i not in random_indices]
    else:
        forget_set = np.random.choice(remain_set,forget_instances_num, replace=False) 
        remain_set = list(set(remain_set) - set(forget_set))
    return remain_set,forget_set

def evaluate_forget_remain_test(model,forget_loader,remain_loader,test_loader,device):
    forget_set_acc = utils.evaluate(model,forget_loader,device)
    print(f"Forget set Accuracy: {forget_set_acc:.2f}%")
    remain_set_acc = utils.evaluate(model,remain_loader,device)
    print(f"Remain set Accuracy: {remain_set_acc:.2f}%")
    test_set_acc = utils.evaluate(model,test_loader,device)
    print(f"Test set Accuracy: {test_set_acc:.2f}")

def load_model(path,lr,device):
    model = torch.load(path)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    return model,optimizer,criterion

# NAIVE  UNLEARNING
def naive_unlearning(architecture,n_inputs,n_classes,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs,results_dict,seed):
    print("\nNaive Unlearning:")
    print("\n")
    utils.set_seed(seed)
    naive_model,optimizer_nu,criterion = utils.initialise_model(architecture,n_inputs,n_classes,device)
    train_naive = Trainer(naive_model, remain_loader, remain_eval_loader, test_loader, optimizer_nu, criterion, device, n_epochs,n_classes,seed)
    naive_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,best_epoch = train_naive.train()
    forget_accuracy,forget_loss,forget_ece = train_naive.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
    print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
    print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")
    results_dict['Naive Unlearning'] = [best_epoch,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]

    return naive_model,results_dict


# GRADIENT ASCENT UNLEARNING

def gradient_ascent(path,remain_loader,remain_eval_loader,test_loader,forget_loader,device,n_epoch_impair,n_epoch_repair,results_dict,n_classes,forget_instances_num,dataset_pointer,seed):
    print("\nGradient Ascent Unlearning:")
    print("\n")
    utils.set_seed(seed)
    if dataset_pointer == 'CIFAR10':
        ga_model,optimizer_ga,criterion = load_model(path,0.1,device)
    else:
        ga_model,optimizer_ga,criterion = load_model(path,(0.01*(256/forget_instances_num)),device)
    evaluate_forget_remain_test(ga_model,forget_loader,remain_loader,test_loader,device)
    ga_train = Unlearner(ga_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_ga, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed)
    ga_model = ga_train.gradient_ascent()

    print("\nFine tuning gradient ascent model:")
    optimizer_ft,criterion = utils.set_hyperparameters(ga_model,lr=0.01)
    ga_fine_tune = Unlearner(ga_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_ft, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed)
    ga_model, remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss, test_ece,best_epoch= ga_fine_tune.fine_tune()
    forget_accuracy,forget_loss,forget_ece = ga_fine_tune.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
    print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
    print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")
    results_dict['Gradient Ascent Unlearning'] = [best_epoch,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return ga_model,results_dict

# FINE TUNE UNLEARNING

def fine_tuning_unlearning(path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs,results_dict,n_classes,seed):
   print("\nFine Tuning Unlearning:")
   utils.set_seed(seed)
   ft_model,optimizer_ft,criterion = load_model(path,0.01,device)
   ft_train = Unlearner(ft_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_ft, criterion, device,0,n_epochs,n_classes,seed)
   ft_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,best_epoch = ft_train.fine_tune()
   forget_accuracy,forget_loss,forget_ece= ft_train.evaluate(forget_loader)
   print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
   print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
   print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")
   results_dict['Fine Tune Unlearning'] = [best_epoch,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
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
  orignial_model,optimizer_gt,criterion,= load_model(path,0.0005,device)

  stochastic_teacher,stochastic_teacher_optimizer,stochastic_teacher_criterion= utils.initialise_model(architecture,n_inputs,n_classes,device,seed)
  evaluate_forget_remain_test(orignial_model,forget_loader,remain_loader,test_loader,device)

  student_model = deepcopy(orignial_model)
  erased_model = train_knowledge_distillation(stochastic_teacher_optimizer,criterion,teacher=stochastic_teacher,student=student_model,train_loader=forget_loader,epochs=n_impair_epochs,T=1,soft_target_loss_weight=1.0,ce_loss_weight=0,device=device)
  print("Stochastic teacher knowledge distillation complete")
  evaluate_forget_remain_test(erased_model,forget_loader,remain_loader,test_loader,device)


  retrained_model = train_knowledge_distillation(optimizer_gt,criterion,teacher=orignial_model,student=erased_model,train_loader=remain_loader,epochs=n_repair_epochs,T=1,soft_target_loss_weight=1.0,ce_loss_weight=0,device=device)
  print("Good teacher knowledge distillation complete")

  forget_accuracy,forget_loss,forget_ece  = retrained_forget_acc = utils.evaluate_test(retrained_model,forget_loader,criterion,n_classes,device)
  remain_accuracy,remain_loss,remain_ece = utils.evaluate_test(retrained_model,remain_loader,criterion,n_classes,device)
  test_accuracy,test_loss,test_ece = utils.evaluate_test(retrained_model,test_loader,criterion,n_classes,device)
  print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
  print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
  print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")

  results_dict['Stochastic Teacher Unlearning'] = [remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
  return retrained_model,results_dict

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
    omp_model,opimizer,criterion,= load_model(path,0.01,device)
    omp_model = global_prune_with_masks(omp_model,pruning_ratio)
    optimizer_omp,criterion = utils.set_hyperparameters(omp_model,lr=0.01)
    print("Pruning Complete:")
    evaluate_forget_remain_test(omp_model,forget_loader,remain_loader,test_loader,device)
    print("\nFine tuning pruned model:")
    omp_train = Unlearner(omp_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_omp, criterion, device,0,n_epochs,n_classes,seed)
    omp_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss, test_ece,best_epoch= omp_train.fine_tune()
    forget_accuracy,forget_loss,forget_ece = omp_train.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
    print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
    print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")
    results_dict["OMP Unlearning"] = [best_epoch,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return omp_model,results_dict

  # CONSINE OMP PRUNE UNLEARNING

def vectorise_model(model):
    """Convert Paramaters to Vector form."""
    return Params2Vec(model.parameters())

def cosine_similarity(base_weights, model_weights):
    """Calculate the cosine similairty between two vectors """
    return torch.nan_to_num(torch.clip(torch.dot(
        base_weights, model_weights
    ) / (
        torch.linalg.norm(base_weights)
        * torch.linalg.norm(model_weights)
    ),-1, 1),0)


def global_prune_without_masks(model, amount):
    """Global Unstructured Pruning of model."""
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                parameters_to_prune.append((mod, "bias"))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for mod in model.modules():
        if hasattr(mod, "weight_orig"):
            if isinstance(mod.weight_orig, torch.nn.Parameter):
                prune.remove(mod, "weight")
        if hasattr(mod, "bias_orig"):
            if isinstance(mod.bias_orig, torch.nn.Parameter):
                prune.remove(mod, "bias")
    return model

def global_prune_with_masks(model, amount):
    """Global Unstructured Pruning of model."""
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                parameters_to_prune.append((mod, "bias"))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model

def kurtosis_of_kurtoses(model):
  kurtosis = []
  for mod in model.modules():
      if hasattr(mod, "weight"):
          if isinstance(mod.weight, torch.nn.Parameter):
              kurtosis.append(stats.kurtosis(mod.weight.cpu().detach().numpy().flatten(), fisher=False))
      if hasattr(mod, "bias"):
          if isinstance(mod.bias, torch.nn.Parameter):
              kurtosis.append(stats.kurtosis(mod.bias.cpu().detach().numpy().flatten(),  fisher=False))
  kurtosis_kurtosis = stats.kurtosis(kurtosis, fisher=False)
  return kurtosis_kurtosis

def cosine_unlearning(path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs,results_dict,n_classes,seed):
    print("\Consine Unlearning:")
    print("\n")
    prune_rate = torch.linspace(0,1,101)
    cosine_sim = []
    base_model,optimizer,criterion,= load_model(path,0.01,device)
    base_vec = vectorise_model(base_model)
    evaluate_forget_remain_test(base_model,forget_loader,remain_loader,test_loader,device)
    for pruning_ratio in prune_rate:
        pruning_ratio = float(pruning_ratio)
        prune_model,optimizer,criterion,= load_model(path,0.01,device)
        prune_model = global_prune_without_masks(prune_model,pruning_ratio)
        prune_model_vec = vectorise_model(prune_model)
        cosine_sim.append(cosine_similarity(base_vec, prune_model_vec).item())
 
    c = torch.vstack((torch.Tensor(cosine_sim), prune_rate))
    d = c.T
    dists = []
    for i in d:
        dists.append(torch.dist(i, torch.Tensor([1, 1])))
    min = torch.argmin(torch.Tensor(dists))

    # kurtosis_of_kurtoses_model = kurtosis_of_kurtoses(base_model)
    # if kurtosis_of_kurtoses_model < torch.exp(torch.Tensor([1])):
    #     prune_modifier = 1/torch.log2(torch.Tensor([kurtosis_of_kurtoses_model]))
    # else:
    #     prune_modifier = 1/torch.log(torch.Tensor([kurtosis_of_kurtoses_model]))
    # safe_prune = prune_rate[min]*prune_modifier.item()

    print(f"Percentage Prune: {prune_rate[min]:.2f}")
    consine_model = global_prune_without_masks(base_model, float(prune_rate[min]))

    print(f"\nModel accuracies post consine pruning:")
    evaluate_forget_remain_test(consine_model,forget_loader,remain_loader,test_loader,device)
    print("\nFine tuning cosine model:")
    optimizer_cosine,criterion = utils.set_hyperparameters(consine_model,lr=0.01)
    cosine_train = Unlearner(consine_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_cosine, criterion, device,0,n_epochs,n_classes,seed)
    consine_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss, test_ece,best_epoch= cosine_train.fine_tune()
    forget_accuracy,forget_loss,forget_ece = cosine_train.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
    print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
    print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")
    results_dict["Cosine Unlearning"] = [best_epoch,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return consine_model,results_dict

def kurtosis_of_kurtoses_unlearning(path,device,remain_loader,remain_eval_loader,test_loader,forget_loader,n_epochs,results_dict,n_classes,seed):
    print("\Consine Unlearning:")
    print("\n")
    prune_rate = torch.linspace(0,1,101)
    cosine_sim = []
    base_model,optimizer,criterion,= load_model(path,0.01,device)
    base_vec = vectorise_model(base_model)
    evaluate_forget_remain_test(base_model,forget_loader,remain_loader,test_loader,device)
    for pruning_ratio in prune_rate:
        pruning_ratio = float(pruning_ratio)
        prune_model,optimizer,criterion,= load_model(path,0.01,device)
        prune_model = global_prune_without_masks(prune_model,pruning_ratio)
        prune_model_vec = vectorise_model(prune_model)
        cosine_sim.append(cosine_similarity(base_vec, prune_model_vec).item())
 
    c = torch.vstack((torch.Tensor(cosine_sim), prune_rate))
    d = c.T
    dists = []
    for i in d:
        dists.append(torch.dist(i, torch.Tensor([1, 1])))
    min = torch.argmin(torch.Tensor(dists))

    kurtosis_of_kurtoses_model = kurtosis_of_kurtoses(base_model)
    if kurtosis_of_kurtoses_model < torch.exp(torch.Tensor([1])):
        prune_modifier = 1/torch.log2(torch.Tensor([kurtosis_of_kurtoses_model]))
    else:
        prune_modifier = 1/torch.log(torch.Tensor([kurtosis_of_kurtoses_model]))
    unsafe_prune = prune_rate[min]+0.1

    print(f"Percentage Prune: {unsafe_prune:.2f}")
    kk_model = global_prune_without_masks(base_model, float(unsafe_prune))

    print(f"\nModel accuracies post consine pruning:")
    evaluate_forget_remain_test(kk_model,forget_loader,remain_loader,test_loader,device)
    print("\nFine tuning cosine model:")
    optimizer_cosine,criterion = utils.set_hyperparameters(kk_model,lr=0.01)
    kk_train = Unlearner(kk_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_cosine, criterion, device,0,n_epochs,n_classes,seed)
    kk_model,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss, test_ece,best_epoch= kk_train.fine_tune()
    forget_accuracy,forget_loss,forget_ece = kk_train.evaluate(forget_loader)
    print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
    print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
    print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")
    results_dict["Kurtosis Unlearning"] = [best_epoch,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return kk_model,results_dict

# Random Label Unlearning - Know as "Unlearning" from Amnesiac Machine Learning paper

def randl_unlearning(path,remain_loader,remain_eval_loader,test_loader,forget_loader,forget_rand_lables_loader,device,n_epoch_impair,n_epoch_repair,results_dict,n_classes,seed):
    randl_model,optimizer_ft,criterion = load_model(path,0.01,device)
    randl_train = Unlearner(randl_model,remain_loader, remain_eval_loader, forget_rand_lables_loader,test_loader, optimizer_ft, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed)
    randl_model,rand_forget_accuracy,rand_forget_loss,rand_forget_ece,test_accuracy,test_loss, test_ece=  randl_train.amnesiac()
    print("Performed Amnesiac Unlearning")
    evaluate_forget_remain_test(randl_model,forget_loader,remain_loader,test_loader,device)

    print("\nFine tuning amnesiac model:")
    optimizer_ft,criterion = utils.set_hyperparameters(randl_model,lr=0.01)
    randl_fine_tune = Unlearner(randl_model,remain_loader, remain_eval_loader, forget_loader,test_loader, optimizer_ft, criterion, device,n_epoch_impair,n_epoch_repair,n_classes,seed)
    randl_model, remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss, test_ece,best_epoch= randl_fine_tune.fine_tune()
    forget_accuracy,forget_loss,forget_ece = randl_fine_tune.evaluate(forget_loader)
    
    print(f"Forget accuracy:{forget_accuracy:.2f}%\tForget loss:{forget_loss:.2f}\tForget ECE:{forget_ece:.2f}")
    print(f"Remain accuracy:{remain_accuracy:.2f}%\tRemain loss:{remain_loss:.2f}\tRemain ECE:{remain_ece:.2f}")
    print(f"Test accuracy:{test_accuracy:.2f}%\tTest loss:{test_loss:.2f}\tTest ECE:{test_ece:.2f}")
    results_dict['Amnesiac Unlearning'] = [best_epoch,remain_accuracy,remain_loss,remain_ece,test_accuracy,test_loss,test_ece,forget_accuracy,forget_loss,forget_ece]
    return randl_model,results_dict











