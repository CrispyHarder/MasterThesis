## Functions to compute cosine similarity of model parameters, predictions agreement and t-SNE plots

import torch
import torch.nn.functional as F
import numpy as np 
from util.saving import get_state_dict_from_checkpoint,load_resnet_model_from_checkpoint # pylint: disable=import-error
from util.inference import get_prediction # pylint: disable=import-error
from util.average_meter import AverageMeter # pylint: disable=import-error

def get_matrix_of_models(list_to_checkpoints, model_type, comparison_function, **kwargs):
    '''compute a matrix for the different comparison values of comparison function
    Args:
        list_to_checkpoints(list(string)): a list of paths to the model checkpoints
        model_type(str): the architecture of the models, from ResNet.resnet 
        comparison_function((model,model)->float)): a function, that takes two models 
            and computes a value
        
    Returns(nd.array): A symmetric matrix containing the output values for every tupel '''
    n_models = len(list_to_checkpoints)
    matrix = np.zeros((n_models,n_models))
    list_models = []

    #load the models
    for i in range(n_models):
        model = load_resnet_model_from_checkpoint(list_to_checkpoints[i], model_type)
        list_models.append(model)
    
    #compute the matrix
    for i in range(n_models):
        for j in range(i+1):
            value = comparison_function(list_models[i],list_models[j],**kwargs)
            matrix[i,j] = value 
            matrix[j,i] = value
    return matrix

def get_flattened_params(model):
    '''reads parameters of a model and return them as a flattened array'''
    p_list = [torch.flatten(p) for p in model.parameters()]
    flat = torch.cat(p_list).view(-1,1)
    return flat

def cosine_sim_model_params(model_0,model_1):
    '''recieves two nn.Modules and computes the cosine 
    similarity of their parameters '''
    params_0 = get_flattened_params(model_0)
    params_1 = get_flattened_params(model_1)
    return F.cosine_similarity(params_0,params_1,dim=0)

def prediction_agreement(model_0,model_1,dataloader,task):
    '''computes the prediction disagreement for two models
    Args:
        model_0(nn.Module): One of the models to compare
        model_1(nn.Module): The other model to compare 
        dataloader(dataloader): The dataloader over which is iterated and on who the 
            predictions are computed and compared
        task(str): either 'class' or 'seg'. The task dictates
            how exactly the disagreement is computed
    
    Returns(float): The prediction disagreement of the two models over the dataloader'''
    agreement = AverageMeter()

    for _,(img,_) in enumerate(dataloader):
        if task == 'class':
            len_batch = len(img)

            if torch.cuda.is_available():
                img = img.cuda()

            # get predictions
            pred_0 = get_prediction(img,model_0,task)
            pred_1 = get_prediction(img,model_1,task)

            # get number of same predictions and then percentage
            same_pred = pred_0.eq(pred_1).float().sum(0)
            agree_percentage = same_pred/len_batch

            #update the averageMeter
            agreement.update(agree_percentage,len_batch)

        if task == 'seg':
            # prob compute dice score of both pred
            raise NotImplementedError
    return agreement.avg

def get_tSNE_plot(list_to_models, model_type, dataloader, number_predictions, task):
    '''computes the t_SNE plot like in the Loss Landscape Paper. Goes through every 
    checkpoint, uses the algorithm from the Loss Landscape paper to map it to 2d.
    Args:
        list_to_checkpoints(list(str)): A list to paths of checkpoints of the models
        dataloader(torch.dataloader): The dataloader to get the samples to predict on
        number_predictions(int): The number of predictions to use for the plot
        task(str): either 'class' or 'seg', which task we are on
    Returns(list(tupels)): A List with every sublist being the 2d projection of a checkpoint 
        of the model
    '''
