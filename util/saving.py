import torch
from ResNet import resnet_cifar10 # pylint: disable=import-error

def save_checkpoint(state, is_best=True, is_checkpoint=True, filename='checkpoint.pth.tar'):
    """
    Save the training model if it has the best val performance
    """
    if is_checkpoint:
        torch.save(state, filename)
    else:
        if is_best:
            torch.save(state, filename)

def get_state_dict_from_checkpoint(checkpoint_path):
    '''loads the state dict from a given checkpoint path'''
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['state_dict']

def load_model_from_checkpoint(checkpoint_path, model_type, dataset_name):
    '''Gets a path to a checkpoint and a model type and loads the model 
    using the state dict'''
    if dataset_name == 'c10':
        model = resnet_cifar10.__dict__[model_type]()
        model.load_state_dict(get_state_dict_from_checkpoint(checkpoint_path))
        if torch.cuda.is_available():
            model.cuda()  
        return model
