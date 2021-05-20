import os 
import torch
from torch.utils.data import Dataset

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

        # now load params and some layers and some kernel depths with 0s
        # all kernels are padded to depth 64 
        # and the layers are conv 1(1 x kern 16 x depth 3), layer1(14 x kern 16 x depth 16),
        # layer2(1 x kern 32 x depth 16, 13 x kern 32 x depth 32)/layer3(1x kern 64 x depth 32 , 13 x kern 64 x depth 64) 
        # -> 1+ 7*2*3 = 43 layers, fc layer (Encode ??)
        # This leads to a either a [43 x 64 x 3 x 3, arch] (layer is in position)
        # or a [1 x 64 x 3 x 3, 14 x 64 x 3 x 3, 14 x 64 x 3 x 3, 14 x 64 x 3 x 3, arch] return
        # weights might be easier to extract then thought, by going through state dict, the names of the weights tell a lot 
        # see private file 
        return [torch.zeros(43,64,3,3),arch]


        