# functions to use during training of a model to get parameters and so on

def get_lr(optimizer):
    '''assuming there is only one parameter group, 
    returns the lr of the optimizer
    Args:
        optimizer(torch.optim.optimizer): the optimizer
    
    Returns(float): the learning rate of the optimizer'''
    for param_group in optimizer.param_groups:
        return param_group['lr']
