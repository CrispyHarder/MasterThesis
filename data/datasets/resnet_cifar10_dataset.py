import os 
import torch
from torch.utils.data import Dataset
from util.saving import get_state_dict_from_checkpoint # pylint: disable=import-error

class resnet_cifar10_parameters_dataset(Dataset):
    '''a class for parameters of resnet20,resnet32,resnet44 in order
    to feed into a data loader'''
    
    def __init__(self,path_to_data):
        '''path to data is a path, whose directories contain the 
        directories containing the paths:
        paths_to_data
            resnet20
                run_1
                    model.th <- actual model state dict
                run_2
                ...
            resnet32
                ...
            resnet44
                ...'''

        self.path_to_data = path_to_data
        self.archs = ['resnet20','resnet32','resnet44']
        #check whether data from other model architectures is present
        for path_to_model in os.listdir(self.path_to_data):
            assert(path_to_model in self.archs)
        
        # get a list to the files with with the model states containing the params
        # for every of the 3 models a list
        self.length = 0
        
        self.paths_to_params_by_model = [[],[],[]]
        for i,path_to_arch in enumerate(os.listdir(self.path_to_data)):
            for run in os.listdir(path_to_arch):
                path_to_params = os.path.join(self.path_to_data,path_to_arch,run,'model.th')
                if os.path.exists(path_to_params):
                    self.paths_to_params_by_model[i].append(path_to_params)
                    self.length += 1
        
        self.length_r20 = len(self.paths_to_params_by_model[0])
        self.length_r32 = len(self.paths_to_params_by_model[1])
        self.length_r44 = len(self.paths_to_params_by_model[2])
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # first check which architecture the index belongs to and get path
        if index < self.length_r20:
            arch = 'resnet20'
            path_to_params = self.paths_to_params_by_model[0][index]
        elif index - self.length_r20 < self.length_r32:
            arch = 'resnet32'
            path_to_params = self.paths_to_params_by_model[1][index-self.length_r20]
        else:
            arch = 'resnet44'
            path_to_params = self.paths_to_params_by_model[2][index-self.length_r20-self.length_r32]

        parameters = []

        state_dict = get_state_dict_from_checkpoint(path_to_params)
        for param_tensor in state_dict:
            if 'conv' in param_tensor:
                params = state_dict[param_tensor]
                p_shape = params.shape
                # pad zeros to ending of param vector
                params = torch.cat((params,torch.zeros(p_shape[0],64-p_shape[1],3,3)),dim=1)
                parameters.append(params)
        
        # # add missing layers for smaller resnets 
        # if arch == 'resnet20':
        #     missing_layers = 8
        #     for _ in range(missing_layers):
        #         parameters.insert(7,torch.zeros(16,64,3,3))
        # if arch == 'resnet32':
        #     missing_layers = 4 
        
        return [parameters,arch]
           