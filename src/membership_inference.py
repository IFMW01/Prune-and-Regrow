from sklearn.model_selection import train_test_split
import load_datasets as ld
from load_datasets import WavToMel,WavToSpec
import Trainer
import json
import utils
import audioMNIST 
import ravdess 
from load_datasets import DatasetProcessor
from torch.utils.data import DataLoader
from Trainer import Trainer
import speech_commands
import numpy as np



def create_membership_inference_dataset(all_processed,seed):
  train_set, test_set = train_test_split(all_processed,train_size = 0.5, test_size=0.2, random_state=seed,shuffle=True)
  print(test_set.size)
  return train_set,test_set

def membership_inference_attack(dataset_pointer,architecture,n_input,n_classes,pipeline,device,n_shadow_models,n_shadow_epochs,logit_dir,softmax_dir):
  test_acc = 0 
  test_loss = 0
  if pipeline == 'mel':
    pipeline_on_wav = WavToMel()
  elif pipeline == 'spec':
     pipeline_on_wav = WavToSpec()
  
  if dataset_pointer == 'SpeechCommands':
    # TODO sort out loader for all speech commands
    train, test = speech_commands.create_speechcommands(dataset_pointer,pipeline_on_wav)
    all_processed = np.append(test, train)
  elif dataset_pointer == 'audioMNIST':
       all_processed = audioMNIST.create_audioMNIST(pipeline,pipeline_on_wav,dataset_pointer)
  elif dataset_pointer == 'ravdess':
       all_processed = ravdess.create_ravdess(pipeline,pipeline_on_wav,dataset_pointer) 

  for seed in range(n_shadow_models):
    # if dataset_pointer == 'SpeechCommands':
    #   train_set_mia,test_set_mia = create_membership_inference_dataset(all_processed,seed)
    #   train_loader = ld.trainset_loader(train_set_mia)
    #   train_eval_loader = ld.testset_loader(train_set_mia)
    #   test_loader = ld.testset_loader(test_set_mia)
    if dataset_pointer == 'SpeechCommands' or dataset_pointer == 'audioMNIST' or dataset_pointer == 'ravdess':
       train_set_mia,test_set_mia = create_membership_inference_dataset(all_processed,seed)
       train_data_mia = DatasetProcessor(train_set_mia)
       test_data_mia = DatasetProcessor(test_set_mia)
       train_loader = DataLoader(train_data_mia, batch_size=256, shuffle=True, num_workers=2)
       train_eval_loader = DataLoader(train_data_mia, batch_size=256, shuffle=False, num_workers=2)
       test_loader = DataLoader(test_data_mia, batch_size=256, shuffle=False, num_workers=2)
       
    model,optimizer,criterion = utils.initialise_model(architecture,n_input,n_classes,device)
    trainer = Trainer(model, train_loader, train_eval_loader, test_loader, optimizer, criterion, device, n_shadow_epochs,n_classes,seed)
    mia_model,train_accuracy,train_loss,train_ece,mia_test_accuracy,mia_test_loss,test_ece,best_epoch = trainer.train()
    test_acc += mia_test_accuracy
    test_loss += mia_test_loss
    print(f'test loss {mia_test_loss}')
    mia_logit_df,mia_loss_df = utils.logits(mia_model, train_eval_loader, test_loader,device)
    filename_logit = (f"MAI_logit_{seed}.csv")
    filename_loss = (f"MAI_loss_{seed}.csv")
    mia_logit_df.to_csv(f"{logit_dir}/{filename_logit}", index = False)
    mia_loss_df.to_csv(f"{softmax_dir}/{filename_loss}", index = False)
    print(f"{filename_logit} and {filename_loss} saved")

  print(f"Average attack test accuracy: {(test_acc/n_shadow_models):.4f}")
  print(f"Average attack test loss: {(test_loss/n_shadow_models):.4f}")

def main(config_mia,config_base):
    dataset_pointer = config_base.get("dataset_pointer", None)
    pipeline = config_base.get("pipeline", None)
    architecture = config_base.get("architecture", None)
    n_classes = config_base.get("n_classes", None)
    n_inputs = config_base.get("n_inputs", None)
    n_shadow_models = config_mia.get("n_shadow_models", None)
    n_shadow_epochs = config_mia.get("n_shadow_epochs", None)

    device = utils.get_device()
    save_dir = f'TRAIN/{dataset_pointer}/{architecture}/MIA'
    logit_dir = save_dir + '/Logits'
    utils.create_dir(logit_dir)
    softmax_dir = save_dir + '/Softmax'
    utils.create_dir(softmax_dir)
    membership_inference_attack(dataset_pointer,architecture,n_inputs,n_classes,pipeline,device,n_shadow_models,n_shadow_epochs,logit_dir,softmax_dir)
    print("FIN")

if __name__ == "__main__":
    with open("./configs/base_config.json","r") as b:
        config_base = json.load(b)    
    with open("./configs/mia_config.json", "r") as mia:
        config_mia = json.load(mia)
    main(config_mia,config_base)

