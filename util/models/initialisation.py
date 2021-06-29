import torch 

def initialize_net_layerwise(net, generator, arch):
    '''takes a network and initialises it with samples from the generator'''
    state_dict = net.state_dict()
    layer = 0
    for param_tensor in state_dict:
        #check whether it is a conv layer 
        if 'conv' in param_tensor:
            shape = state_dict[param_tensor].size()
            sampled_layer = generator.sample(num_samples = 1 , 
                current_device = torch.device('cpu'), label = arch)[0] #layer might follow 
            cut_layer = sampled_layer[:shape[0],:shape[1],:,:]
            state_dict[param_tensor] = cut_layer
            layer += 1
    net.load_state_dict(state_dict)
    return net