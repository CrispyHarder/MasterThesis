# for functions after the model computes output
import torch

def get_prediction(input,model,task):
    '''computes the predictions for an input batch and a model
    Args:
        input(torch.tensor): a input batch
        model(nn.Module): a model
        task(str): the task, either 'class' or 'seg'

    Returns(torch.tensor): The predictions for the input batch
    '''
    output = model(input)
    if task == 'class':
        return torch.argmax(output,dim=1)
    if task == 'seg':
        raise NotImplementedError
        