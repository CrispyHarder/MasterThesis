import torch
import json
from models.ResNet import resnet_cifar10 # pylint: disable=import-error

def save_checkpoint(state, is_best=True, is_checkpoint=True, filename='checkpoint.pth.tar'):
    """
    Save the training model if it has the best val performance
    """
    if is_checkpoint:
        torch.save(state, filename)
    else:
        if is_best:
            torch.save(state, filename)

def get_state_dict_from_checkpoint(checkpoint_path, map_location=None):
    '''loads the state dict from a given checkpoint path'''
    if map_location:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    elif torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint['state_dict']

def load_model_from_checkpoint(checkpoint_path, model_type, dataset_name):
    '''Gets a path to a checkpoint and a model type and loads the model 
    using the state dict'''
    if dataset_name == 'cifar10':
        model = resnet_cifar10.__dict__[model_type]()
        model.load_state_dict(get_state_dict_from_checkpoint(checkpoint_path))
        if torch.cuda.is_available():
            model.cuda()  
        return model

def save_training_hparams(args):
    unimportant_list = ['data_storage','device','workers',
    'print_freq','resume','evaluate','pretrained','half','save_dir',
    'save_every','nr_runs','runs_start_at']
    dik = dict()
    for arg in vars(args):
        if arg not in unimportant_list:
            dik[arg] = getattr(args, arg)
    with open('hyperparameters.json','w') as file:
        json.dump(dik,file,indent="")

def save_dict_values(dik,name='results'):
    with open(name+'.json','w') as file:
        json.dump(dik,file, indent="")


