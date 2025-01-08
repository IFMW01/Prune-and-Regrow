import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import json
import Trainer 
from Trainer import Trainer
from datasets_unlearn import load_datasets as ld
import utils 
import argparse


def options_parser():
    parser = argparse.ArgumentParser(description="Arguments for creating model")
    parser.add_argument(
        "--dataset_pointer",
        required=True,
        type=str
    )

    parser.add_argument(
        "--architecture",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--opt",
        required=True,
        type=str
    )

    parser.add_argument(
        "--lr",
        required=False,
        type=float,
        default= 0.001

    )

    parser.add_argument(
        "--n_epochs",
        required=False,
        type=int,
        default= 200
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    args = parser.parse_args()

    return args

def create_base_model(train,save_model_path,save_mia_path,device,seed,train_loader,test_loader,results_dict):
    results_dict[f'{seed}'] = {}
    best_model,train_accuracy,train_loss,train_ece,test_acc,test_loss,test_ece,best_epoch,best_time= train.train()
    torch.save(best_model,f"{save_model_path}Model_{test_acc:.5f}_{test_loss:.5f}.pth")
    df_loss_outputs = utils.logits(best_model,train_loader,test_loader,device)
    df_loss_outputs.to_csv(f'{save_model_path}loss_outputs.csv',index = False)
    df_loss_outputs.to_csv(f'{save_mia_path}{seed}_loss_outputs.csv',index = False)
    results_dict[f'{seed}'] = utils.update_dict(results_dict[f'{seed}'],best_time,best_epoch,train_accuracy,train_loss,train_ece,test_acc,test_loss,test_ece)
    return results_dict

def main(args):
 
    if args.dataset_pointer == 'CIFAR10':
        n_classes = 10
    elif args.dataset_pointer == 'CIFAR100': 
        n_classes = 100
    elif args.dataset_pointer == 'Tiny ImageNet': 
        n_classes = 200

    print("Experiemental setup")
    print(f"Dataset pointer: {args.dataset_pointer}")
    print(f"Architecture: {args.architecture}")
    print(f"Optimizer: {args.opt}")
    print(f"Learning Rate: {args.lr}")
    print(f"Number of epochs: {args.n_epochs}")
    print(f"Seed: {args.seed}")
    print(f"Number of classes: {n_classes}")

    device = utils.get_device()
    device = utils.get_device()
    print("Device Stats")
    print(device)
    results_dict = {}

    # Iterates over the provided seeds and creates model 
    train_loader,train_eval_loader,test_loader = ld.load_datasets(args.dataset_pointer,False)

    save_dir = f"Results/{args.dataset_pointer}/{args.architecture}"
    utils.set_seed(args.seed)
    model,optimizer,criterion = utils.initialise_model(args.architecture,args.opt,n_classes,device)
    utils.create_dir(save_dir)
    save_model_path = f'{save_dir}/{args.seed}/'
    utils.create_dir(save_model_path)
    save_mia_path = f'{save_dir}/MIA/'
    utils.create_dir(save_mia_path)
    train = Trainer(model, train_loader, train_eval_loader, test_loader, optimizer, criterion, device, args.n_epochs,n_classes,args.seed)
    results_dict = create_base_model(train,save_model_path,save_mia_path,device,args.seed,train_loader,test_loader,results_dict)

    print(f'Final of all trained models: {results_dict}')

    with open(f"{save_model_path}/training_results.json",'w') as f:

        json.dump(results_dict,f)

    print("FIN")

if __name__ == "__main__":
    args = options_parser()        
    main(args)