from abc import abstractmethod
import os 
import torch
from torch.utils.data import Dataset
from util.saving import get_state_dict_from_checkpoint # pylint: disable=import-error
from util.data.data_reshaping import stack_to_side, pad_layer, one_hot # pylint: disable=import-error

class Resnet_cifar10_dataset(Dataset):
    '''a parent class for any dataset containing parameters of resnet20,resnet32,resnet44 in order
    to feed into a data loader'''

    def __init__(self,path_to_data,train,use_labels=False):
        '''path to data is a path, whose directories contain the 
        directories containing the paths:
        paths_to_data
            resnet20
                train
                    run_1
                        model.th <- actual model state dict
                    run_2
                ...
            resnet32
                ...
            resnet44
                ...
        Args:
            path_to_data(str): path to the data
            train(bool): whether or train or validation data should be used'''
        super().__init__()
        self.use_labels = use_labels
        self.number_archs = 3
        self.path_to_data = path_to_data
        self.archs = ['resnet20','resnet32','resnet44']
        #check whether data from other model architectures is present
        for path_to_arch in os.listdir(self.path_to_data):
            assert(path_to_arch in self.archs)

        if train:
            self.mode = 'train'
        else:
            self.mode = 'val'

        @abstractmethod
        def __len___(self):
            pass

        @abstractmethod
        def __getitem__(self):
            pass

class Resnet_cifar10_parameters_dataset(Resnet_cifar10_dataset):
    '''a class for parameters of resnet20,resnet32,resnet44 in order
    to feed into a data loader'''
    
    def __init__(self,path_to_data,train):
        '''path to data is a path, whose directories contain the 
        directories containing the paths:
        paths_to_data
            resnet20
                train
                    run_1
                        model.th <- actual model state dict
                    run_2
                ...
            resnet32
                ...
            resnet44
                ...
        Args:
            path_to_data(str): path to the data
            train(bool): whether or train or validation data should be used'''

        super().__init__(path_to_data,train)

        # get a list to the files with with the model states containing the params
        # for every of the 3 models a list
        self.length = 0
        
        self.paths_to_params_by_arch = [[],[],[]]
        for i,arch in enumerate(os.listdir(self.path_to_data)):
            path_to_arch = os.path.join(self.path_to_data,arch,self.mode)
            for run in os.listdir(path_to_arch):
                path_to_params = os.path.join(path_to_arch,run,'model.th')
                if os.path.exists(path_to_params):
                    self.paths_to_params_by_arch[i].append(path_to_params)
                    self.length += 1
        
        self.length_r20 = len(self.paths_to_params_by_arch[0])
        self.length_r32 = len(self.paths_to_params_by_arch[1])
        self.length_r44 = len(self.paths_to_params_by_arch[2])
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # first check which architecture the index belongs to and get path
        if index < self.length_r20:
            arch = 0 # 'resnet20'
            path_to_params = self.paths_to_params_by_arch[0][index]
        elif index - self.length_r20 < self.length_r32:
            arch = 1 # 'resnet32'
            path_to_params = self.paths_to_params_by_arch[1][index-self.length_r20]
        else:
            arch = 2 # 'resnet44'
            path_to_params = self.paths_to_params_by_arch[2][index-self.length_r20-self.length_r32]

        parameters = []
        masks = []

        state_dict = get_state_dict_from_checkpoint(path_to_params,map_location='cpu')
        for param_tensor in state_dict:
            if 'conv' in param_tensor:
                params = state_dict[param_tensor]
                mask = torch.ones_like(params)

                params = pad_layer(params, 64, 64)
                mask = pad_layer(mask, 64, 64)

                params = stack_to_side(params)
                mask = stack_to_side(mask)

                parameters.append(params)
                masks.append(mask)
        arch = one_hot(arch, self.number_archs)
        return torch.stack(parameters), torch.stack(masks), arch # pylint: disable=not-callable


class Resnet_cifar10_layer_parameters_dataset(Resnet_cifar10_dataset):
    '''a class for layer parameters of resnet20,resnet32,resnet44 in order
    to feed into a data loader'''

    def __init__(self, path_to_data,train):
        '''path to data is a path, whose directories contain the 
        directories containing the paths:
        paths_to_data
            resnet20
                train
                    run_1
                        model.th <- actual model state dict
                    run_2
                ...
            resnet32
                ...
            resnet44
                ...
        Args:
            path_to_data(str): path to the data
            train(bool): whether or train or validation data should be used'''
        super().__init__(path_to_data, train)

        # get a list to the files with with the model states containing the params
        # for every of the 3 model archs a list
        self.length = 0
        self.number_layers = 0

        self.paths_to_params_by_arch = [[], [], []]
        for i,arch in enumerate(os.listdir(self.path_to_data)):
            path_to_arch = os.path.join(self.path_to_data, arch,self.mode)
            for run in os.listdir(path_to_arch):
                path_to_params = os.path.join(path_to_arch, run, 'model.th')
                layer_number = 0
                if os.path.exists(path_to_params):
                    state_dict = get_state_dict_from_checkpoint(path_to_params)
                    for param_tensor in state_dict:
                        if 'conv' in param_tensor:
                            self.paths_to_params_by_arch[i].append((path_to_params, layer_number))
                            self.length += 1
                            layer_number += 1 
                            self.number_layers = max(self.number_layers, layer_number)
        
        self.length_r20 = len(self.paths_to_params_by_arch[0])
        self.length_r32 = len(self.paths_to_params_by_arch[1])
        self.length_r44 = len(self.paths_to_params_by_arch[2])

    def __len__(self):
        return self.length

    def __getitem__(self,index):

        # first check which architecture the index belongs to and get path
        if index < self.length_r20:
            arch = 0 # 'resnet20'
            path_to_params, layer_number = self.paths_to_params_by_arch[0][index]
        elif index - self.length_r20 < self.length_r32:
            arch = 1 # 'resnet32'
            path_to_params, layer_number = self.paths_to_params_by_arch[1][index-self.length_r20]
        else:
            arch = 2 # 'resnet44'
            path_to_params, layer_number = self.paths_to_params_by_arch[2][index-self.length_r20-self.length_r32]

        state_dict = get_state_dict_from_checkpoint(path_to_params, map_location='cpu')
        nr_layer = 0
        for param_tensor in state_dict:
            if 'conv' in param_tensor:
                if nr_layer == layer_number:  
                    params = state_dict[param_tensor]
                    mask = torch.ones_like(params)

                    params = pad_layer(params, 64, 64)
                    mask = pad_layer(mask, 64, 64)

                    params = stack_to_side(params)
                    mask = stack_to_side(mask)
                    arch = one_hot(arch,self.number_archs)
                    layer = one_hot(nr_layer,self.number_layers)
                    return params, mask, arch, layer 
                else: 
                    nr_layer += 1

class Resnet_cifar10_kernel_parameters_dataset(Resnet_cifar10_dataset):
    '''a class for slice-wise parameters of resnet20, resnet32, resnet44 in order
    to feed into a data loader'''

    def __init__(self, path_to_data,train):
        '''path to data is a path, whose directories contain the 
        directories containing the paths:
        paths_to_data
            resnet20
                train
                    run_1
                        model.th <- actual model state dict
                    run_2
                ...
            resnet32
                ...
            resnet44
                ...
        Args:
            path_to_data(str): path to the data
            train(bool): whether or train or validation data should be used'''
        super().__init__(path_to_data, train)

        # get a list to the files with with the model states containing the params
        # for every of the 3 model archs a list
        self.length = 0
        self.number_layers = 0
        self.number_kernels = 0

        self.params = []
        for arch in os.listdir(self.path_to_data):
            path_to_arch = os.path.join(self.path_to_data, arch, self.mode)
            for run in os.listdir(path_to_arch):
                path_to_params = os.path.join(path_to_arch, run, 'model.th')
                if os.path.exists(path_to_params):
                    state_dict = get_state_dict_from_checkpoint(path_to_params)
                    layer_number = 0
                    for param_tensor in state_dict:
                        if 'conv' in param_tensor:
                            for kernel_number, kernel in enumerate(state_dict[param_tensor]):
                                self.params.append((path_to_params,
                                                    arch,
                                                    layer_number,
                                                    kernel_number))
                                self.length += 1
                            self.number_kernels = max(self.number_kernels, kernel_number)
                            layer_number += 1 
                    self.number_layers = max(self.number_layers, layer_number)

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        
        names_to_numbers = {'resnet20':0, 
                            'resnet32':1,
                            'resnet44':2}
        path, arch, nr_layer, nr_kernel = self.params[index]
        arch = names_to_numbers[arch]
        state_dict = get_state_dict_from_checkpoint(path)
        layer_number = 0
        for param_tensor in state_dict:
            if 'conv' in param_tensor:
                if layer_number == nr_layer:
                    for kernel_number, kernel in enumerate(state_dict[param_tensor]):
                        if kernel_number == nr_kernel:
                            return kernel, arch, nr_layer
                layer_number += 1 


class Resnet_cifar10_slice_parameters_dataset(Resnet_cifar10_dataset):
    '''a class for slice-wise parameters of resnet20, resnet32, resnet44 in order
    to feed into a data loader'''

    def __init__(self, path_to_data,train):
        '''path to data is a path, whose directories contain the 
        directories containing the paths:
        paths_to_data
            resnet20
                train
                    run_1
                        model.th <- actual model state dict
                    run_2
                ...
            resnet32
                ...
            resnet44
                ...
        Args:
            path_to_data(str): path to the data
            train(bool): whether or train or validation data should be used'''
        super().__init__(path_to_data, train)

        # get a list to the files with with the model states containing the params
        # for every of the 3 model archs a list
        self.length = 0
        self.number_layers = 0
        self.number_kernels = 0
        self.number_slices = 0

        self.params = []
        for arch in os.listdir(self.path_to_data):
            path_to_arch = os.path.join(self.path_to_data, arch, self.mode)
            for run in os.listdir(path_to_arch):
                path_to_params = os.path.join(path_to_arch, run, 'model.th')
                if os.path.exists(path_to_params):
                    state_dict = get_state_dict_from_checkpoint(path_to_params)
                    layer_number = 0
                    for param_tensor in state_dict:
                        if 'conv' in param_tensor:
                            for kernel_number, kernel in enumerate(state_dict[param_tensor]):
                                for slice_number, slice in enumerate(kernel):
                                    self.params.append((path_to_params,
                                                    arch,
                                                    layer_number,
                                                    kernel_number,
                                                    slice_number))
                                    self.length += 1
                                self.number_slices = max(self.number_slices, slice_number)
                            self.number_kernels = max(self.number_kernels, kernel_number)
                            layer_number += 1 
                    self.number_layers = max(self.number_layers, layer_number)

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        
        names_to_numbers = {'resnet20':0, 
                            'resnet32':1,
                            'resnet44':2}
        path, arch, nr_layer, nr_kernel, nr_slice = self.params[index]
        arch = names_to_numbers[arch]
        state_dict = get_state_dict_from_checkpoint(path)
        layer_number = 0
        for param_tensor in state_dict:
            if 'conv' in param_tensor:
                if layer_number == nr_layer:
                    for kernel_number, kernel in enumerate(state_dict[param_tensor]):
                        if kernel_number == nr_kernel:
                            for slice_number, slice in enumerate(kernel):
                                if slice_number == nr_slice:
                                    return slice, arch, nr_layer
                layer_number += 1 
        





