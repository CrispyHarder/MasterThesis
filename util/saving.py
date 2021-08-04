import importlib
import torch
import sys
import os 
import json

from torch.functional import atleast_1d
from models.ResNet.cifar10 import resnet # pylint: disable=import-error
from pathlib import PurePath
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
        model = resnet.__dict__[model_type]()
        model.load_state_dict(get_state_dict_from_checkpoint(checkpoint_path))
        if torch.cuda.is_available():
            model.cuda()  
        return model

def save_training_hparams(args,path,add_hparams={}):
    save_path = os.path.join(path,'hyperparameters.json')
    unimportant_list = ['data_storage','device','workers',
    'print_freq','resume','evaluate','pretrained','half','save_dir',
    'save_every','nr_runs','runs_start_at']
    dik = dict()
    for arg in vars(args):
        if arg not in unimportant_list:
            dik[arg] = getattr(args, arg)
    for key in add_hparams.keys():
        dik[key] = add_hparams[key]
    with open(save_path,'w') as file:
        json.dump(dik,file,indent="")

def save_dict_values(dik,path, name='results'):
    save_path = os.path.join(path,name)
    with open(save_path+'.json','w') as file:
        json.dump(dik,file, indent="")

def load_model_from_path(path):
    if 'resnet_cifar10' in path:
        # get basic path 
        path_to_class = os.path.join('models','parameter_learners')

        #make dicts to translate the path names to class name
        path_to_class_dict = [
            {
                'layer':'Layer'
            },
            {
                True:'C',
                False:''
            },
            {
                'VAE':'VAE',
                'VQVAE':'VQVAE'
            },
            {
                'resnet_cifar10':'resC10'
            }]

        #get the modifications for the class/generator
        path_parts = PurePath(path).parts
        basic_model = path_parts[-5]
        dataset = path_parts[-4]
        wise = path_parts[-3]
        version = path_parts[-2]
        # account for baseline_conditional
        version = version.split('_')[0]
        hp_path = os.path.join(path,'hyperparameters.json')
        with open(hp_path,'r') as file:
            hyper_dict = json.load(file)
        conditional = hyper_dict['conditional']

        # get the name of the class for the model to be loaded 
        architecture = ''
        keywords = [wise,conditional,basic_model,dataset]
        for dik,key in zip(path_to_class_dict,keywords):
            architecture += dik[key]
        
        #modify the path to class
        
        path_to_class = os.path.join(path_to_class,dataset,wise)
        sys.path.append(path_to_class)
        architectures = importlib.import_module(version)
        model = getattr(architectures,architecture)(**hyper_dict)
        sys.path.remove(path_to_class)

        # get the state dict of the model 
        cp_path = os.path.join(path,'model.th')
        state_dict = get_state_dict_from_checkpoint(cp_path)
        model.load_state_dict(state_dict)
    
    elif 'cifar10' in path and 'ResNet' in path:
        #then it has to be a ResNet

        #set initial path
        path_to_class = os.path.join('models','ResNet')
    
        #get the modifications for the class/generator
        path_parts = PurePath(path).parts
        basic_model = path_parts[-4]
        dataset = path_parts[-3]
        arch = path_parts[-2]

        # get the name of the class for the model to be loaded 
        architecture = arch
        
        #modify the path to class
        path_to_class = os.path.join(path_to_class,dataset)
        sys.path.append(path_to_class)
        architectures = importlib.import_module('resnet')
        model = getattr(architectures,architecture)()
        sys.path.remove(path_to_class)

        # get the state dict of the model 
        cp_path = os.path.join(path,'model.th')
        state_dict = get_state_dict_from_checkpoint(cp_path)
        model.load_state_dict(state_dict)

    return model
    
    

    

    


