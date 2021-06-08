import unittest
import os 
import torch
from torch.utils.data.dataloader import DataLoader 

class DatasetTests(unittest.TestCase):

    def test_resnet_cifar10_parameters(self):
        from data.datasets.resnet_cifar10_dataset import Resnet_cifar10_parameters_dataset # pylint: disable=all
        path = os.path.join('storage','models','ResNet','cifar10')
        ds = Resnet_cifar10_parameters_dataset(path,train=True) 
        for i in range(ds.__len__()):
            ds.__getitem__(i)

        print('Resnet_cifar10_parameters working')

    def test_resnet_cifar10_layer_parameters(self):
        from data.datasets.resnet_cifar10_dataset import Resnet_cifar10_layer_parameters_dataset # pylint: disable=all
        path = os.path.join('storage','models','ResNet','cifar10')
        ds = Resnet_cifar10_layer_parameters_dataset(path,train=True) 
        for i in range(ds.__len__()):
            ds.__getitem__(i)

        print('Resnet_cifar10_layer_parameters working')

    def test_all(self):
        self.test_resnet_cifar10_parameters()
        self.test_resnet_cifar10_layer_parameters()

class DataloaderTests(unittest.TestCase):

    def __init__(self):
        super().__init__()
        from torch.utils.data import DataLoader
        self.cuda_avail = torch.cuda.is_available()
        self.num_workers = 4 if self.cuda_avail else 0
        if self.cuda_avail:
            torch.multiprocessing.set_start_method('spawn')

    def test_resnet_cifar10_parameters(self):
        from data.datasets.resnet_cifar10_dataset import Resnet_cifar10_parameters_dataset # pylint: disable=all
        path = os.path.join('storage','models','ResNet','cifar10')
        ds = Resnet_cifar10_parameters_dataset(path,train=True) 
        dl = DataLoader(ds,2,num_workers=self.num_workers,pin_memory=True,shuffle=True)
        for output in enumerate(dl):
            pass 
        print('test_resnet_cifar10_parameters dataloader working')

    def test_resnet_cifar10_layer_parameters(self):
        from data.datasets.resnet_cifar10_dataset import Resnet_cifar10_layer_parameters_dataset # pylint: disable=all
        path = os.path.join('storage','models','ResNet','cifar10')
        ds = Resnet_cifar10_layer_parameters_dataset(path,train=True) 
        dl = DataLoader(ds,2,num_workers=self.num_workers,pin_memory=True,shuffle=True)
        for output in enumerate(dl):
            pass 
        print('test_resnet_cifar10_layer_parameters dataloader working')
    
    def test_all(self):
        self.test_resnet_cifar10_parameters()
        self.test_resnet_cifar10_layer_parameters()





        