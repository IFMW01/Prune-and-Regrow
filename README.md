# Machine Unlearning In Audio Analysis For Data Privacy

This code repository contains the implementation for the M.Phil. research project "Machine Unlearning Analysis In Audio For Data Privacy". The code repository focuses on implementing unlearning methods currently designed for the computer vision and audio domain to audio so that unlearning requests can be satisfied in the audio. Alongside this novel verification mechanisms are being explored in tandem with novel machine unlearning approaches tailored for audio data. The comprehensive study covers two datasets and three architectures to provide a comprehensive overview of the current state of unlearning in audio and the potential challenges that exist. All results presented in the thesis are averaged over the five seeds listed in the base_config.json in the configs folder. There is support for spectrogram and mel spectrogram representations in this setup. However, only mel spectrograms are explored for the thesis. 

First run:

pip install -r requirements.txt

## 1. Create base model

To run the experiments, first, go to the config repository and modify base_config.json:
- "dataset_pointer": "audioMNIST" or "SpeechCommands".
- "pipeline": "mel"  (to recreate experiments presented in the thesis, although "spec" is supported for standard spectrograms)
- "architecture": "VGGishMel" or "CTCmel" or "ViTmel" (for "spec" is supported for standard spectrograms:"VGGishSpec" or "CTCspec" or "ViTspec")
- "n_epochs": 30
- "seeds": [40,41,42,43,44]
- "n_classes":10 (for audioMNIST) or 35 (for SpeechCommands)
- "n_inputs":1

Once the config is correctly modified, then run the following:

**python train_base.py**

This will trigger the training sequence for the base model across the seeds. First, the data is downloaded and converted into the correct format, and then the training sequences commence, and the model, loss outputs on train and test and overall results are saved for each model on each seed.

**The following data repositories and files are created:**

- ./>"pipeline" - that contains the train and test set lists
- ./>"pipeline">"dataset_pointer">file.pth - contains the data converted into the   "pipeline" implemented to speed up training time
  
- ./>Results>"dataset_pointer">"architecture": This repository holds the results.
- ./>Results>"dataset_pointer">"architecture">"seed": This folder holds the saved              base model.

    Files held in this repository:
      -  Model_test_acc_loss.pth: Each model trained on each seed is saved.
      -  loss_outputs.csv: Train and Test set loss outputs are saved
This happens for all seeds in the list "seeds".

-./>Results>"dataset_pointer">"architecture">training_results.json: Holds the         dictionary for all the results of the base models on the train and test set for each     seed.

- ./>Results>"dataset_pointer">"architecture">MIA>seed_loss_outputs.csv for each model         created on each seed, the loss outputs on train and test are saved within the CSV file       seed_loss_outputs.csv.

## 2. Create attack model
To run the experiments and increase the number of attack models, go to attack_config.json and modify "n_attack_models" The default for results in the thesis is 3.

Once the config is correctly modified, then run the following:

**python attack.py**

This will begin the training sequence for the attack model. First, it creates a balanced dataset from the seed_loss_outputs.csv for each, ensuring a 50% train and 50% test. Then, the attack models are trained and created from this dataset.

**The following data repositories and files are created:**
- ./>Results>"dataset_pointer">"architecture">MIA_all_balanced.csv: This contains the dataset used to train the attack models. It is composed of all the models trained on each seed loss output and is a balanced dataset of train and test loss outputs.
- ./>Results>"dataset_pointer">"architecture">Attack: respository containing attack models
  Files held in this repository:
    - attack_model_0.pth: saved attack model
    - attack_model_1.pth: saved attack model
    - attack_model_2.pth: saved attack model
  
- ./>Results>"dataset_pointer">"architecture">Attack>attack_model_results.json: Each       attack models accuracy results.

## 3. Perform unlearning

To run the experiments, first, go to the config repository and modify **only these values** in unlearn_config.json:

- "forget_random": "true" or "false"
- "forget_percentage": 10 or 20 or 30 (for experiments presented in the thesis)
- "forget_classes": "true" or "false" (cannot be "true" if forget_random is "true")
- "forget_classes_num" 1 or 2 or 3

Each test is run independently so for forget random for 10% or 20% or 30% you need to run this command: 

**python unlearn.py**

For each forget_percentage, modify the config before running the command each time. The same is true for forget_classes to run tests for 1 or 2 or 3 class removal. 

The script above will initiate the unlearning process. First, a forget and remain set is generated. Then, it gets the base model trained for each seed and performs the unlearning methods in the following order: Naive, Gradient Ascent, Stochastic Teacher, OMP, Cosine, Post Optimal Prune (POP), Amnesiac, and Label Smoothing. The results are saved in the last seed removal directory.

**The following data repositories and files are created:**

- ./>Results>"dataset_pointer">"architecture">UNLEARN>Item_Removal: if forget_random is true this      holds data for item removal
  
- ./>Results>"dataset_pointer">"architecture">UNLEARN>Class_Removal:  if forget_classes is true this   holds data for class removal

**The following files and repositories are created in the respective removal folder depending on what time of removal is being performed (Item or Class), where amount removed   is the number of instances held in teh forget set.**

- ./>Results>"dataset_pointer">"architecture">UNLEARN>Removal_Type>amount_removed>Seed:         Holds the removal data for each seed to perform unlearning on from the base model
   Files held in this repository:
      - Naive.pth: The Naive retained model
      - unlearning_method_name_loss_forget.csv : The loss values on the forget set for each         unlearning method and Naive and original model
      - unlearning_method_name_loss_remain.csv : The loss values on the remain set for each         unlearning method and Naive and original model
      - unlearning_method_name_loss_test.csv : The loss values on the test set for each              unlearning method and Naive and original model
  
    - unlearning_results.json: The unlearning results stored in the last seed repository          created

Following this all results can be extracted and analysed. 
  
  









