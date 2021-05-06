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

def get_prediction_on_data(model,dataloader,number_pred,task,return_labels=False):
    '''computes a number of predictions on the dataloader and returns 
    a tensor of predictions in the class case or a list of segmented slices 
    in the seg case
    Args:
        model: the model
        dataloader(torch.dataloader)
        number_pred(int): the maximum number of predictions to compute. if 0 all data it used
        task: either 'class' or 'seg' 

    Returns(tensor): with the predictions'''
    number_samples = 0
    predictions = []
    labels = []
    for _,(img,label) in enumerate(dataloader):
        
        if  not number_pred==0 and number_samples > number_pred:
            break
        number_samples += len(img)
        
        if torch.cuda.is_available():
            img = img.cuda()
        
        batch_pred = get_prediction(img,model,task)
        
        predictions.append(batch_pred)
        if return_labels:
            labels.append(label)

    if task == 'class':
        predictions = torch.stack(predictions) # pylint: disable=not-callable
        predictions = torch.flatten(predictions) 
        if return_labels:
            labels = torch.stack(labels)
            labels = torch.flatten(labels)
    
    if task == 'seg':
        raise NotImplementedError
    
    return predictions,labels

        