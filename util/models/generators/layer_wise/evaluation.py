import itertools 
import torch
import numpy as np 
from util.saving import load_model_from_path
from util.data.data_reshaping import one_hot

def check_mean_std_generated(gen_path, number_of_samples=5):
    generator = load_model_from_path(gen_path)
    conditional = hasattr(generator,'cond_attributes')
    if conditional:
        cond_attributes = getattr(generator,'cond_attributes')
        possible_labels = [np.arange(0,num) for num in cond_attributes]
        possible_combinations = list(itertools.product(*possible_labels))
        for combination in possible_combinations:
            oh_comb = [one_hot(combination[i],cond_attributes[i]) for i in len(combination)]
            samples = generator.sample(number_of_samples,torch.device('cpu'),*oh_comb)
            for sample in samples:
                mean = torch.mean(sample)
                std = torch.std(sample)
                print('Sample of labels {} has a mean of {:.3f} and a std of {:.3f}'. format(
                        combination,mean,std))
    else: 
        samples = generator.sample(number_of_samples,torch.device('cpu'))
        for i,sample in enumerate(samples):
            print(samples[i,0,:3,:3])
            mean = torch.mean(sample)
            std = torch.std(sample)
            print('Sample has a mean of {:.3f} and a std of {:.3f}'. format(mean,std))
        
        
        

