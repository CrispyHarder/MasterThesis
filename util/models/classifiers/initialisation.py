import torch 
from util.data.data_reshaping import one_hot, side_to_stack

def initialize_net_layerwise(net, generator, arch_label=0, number_archs=100, number_layers=100):
    '''takes a network and returns a state dict for it with sampled parameters for 
    all convolutional layers'''
    state_dict = net.state_dict()
    layer = 0
    for param_tensor in state_dict:
        #check whether it is a conv layer 
        if 'conv' in param_tensor:
            shape = state_dict[param_tensor].size()
            sampled_layer = generator.sample(num_samples = 1, 
                                current_device = torch.device('cpu'),
                                arch = torch.unsqueeze(one_hot(arch_label,number_archs),0), 
                                layer = torch.unsqueeze(one_hot(layer,number_layers),0))[0] 
            sampled_layer = side_to_stack(sampled_layer,k_size=3)
            cut_layer = sampled_layer[:shape[0],:shape[1],:,:]
            state_dict[param_tensor] = cut_layer
            layer += 1
    return state_dict