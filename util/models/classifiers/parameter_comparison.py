## Functions to compute cosine similarity of model parameters, predictions agreement and t-SNE plots

import torch
import torch.nn.functional as F
import numpy as np 
import os
import re
from util.saving import load_model_from_checkpoint# pylint: disable=import-error
from util.models.classifiers.inference import get_prediction_on_data # pylint: disable=import-error
from sklearn.manifold import TSNE

def get_matrix_of_models(list_to_checkpoints, model_type, dataset_name, comparison_function, **kwargs):
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
        model = load_model_from_checkpoint(list_to_checkpoints[i], model_type, dataset_name)
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

def prediction_disagreement(pred_0,pred_1,task,only_on_right_pred=False):
    '''computes the prediction disagreement between two prediction tensors
    Args:
        pred_0(torch.tensor): One of the models to compare
        pred_1(torch.tessor): The other model to compare 
        task(str): either 'class' or 'seg'. The task dictates
            how exactly the disagreement is computed
        only_on_right_pred(bool):whether the prediction disagreement is only to be 
            computed on predictions where at least one model predicts correctly

    Returns(float): The prediction agreement of the two tensors'''

    if task == 'class':
        if not only_on_right_pred:
            # default behaviour
            len_pred = len(pred_0)

            # get number of same predictions and then percentage
            same_pred = pred_0.eq(pred_1).float().sum(0)
            agree_percentage = same_pred/len_pred
        else: 
            #get the predictions and labels
            pred0 = pred_0[0]
            label0 = pred_0[1]
            pred1 = pred_1[0]
            label1 = pred_1[1]

            #get the predictions where at least one model was right
            equal_0 = pred0.eq(label0)
            equal_1 = pred1.eq(label1)
            at_least_one_right = torch.logical_or(equal_0,equal_1)
            nr_at_least_one_right = at_least_one_right.float().sum(0)

            # get the number of same predictions, where also at least one prediction was right
            same_pred_at_least_one_right = torch.dot(pred0.eq(pred1).float(),at_least_one_right.float())

            agree_percentage = same_pred_at_least_one_right/nr_at_least_one_right


    if task == 'seg':
        # prob compute dice score of both pred
        raise NotImplementedError
    return 1-agree_percentage

def get_prediction_disagreement_matrix(list_to_checkpoints,model_type,dataset_name,dataloader,task,only_on_right_pred=False):
    '''computes the pred_agreement_matrix in a effient way for more then 2 models
    Args:
        list_to_checkpoints(list(str)): A list to paths of checkpoints of the models
        model_type(str): The type of the model
        dataset_name(str): The name of the dataset; together with model type identifies the model
        dataloader(torch.dataloader): The dataloader to get the samples to predict on
        task(str): either 'class' or 'seg', which task we are on 
        only_on_right_pred(bool):whether the prediction disagreement is only to be 
            computed on predictions where at least one model predicts correctly

    Returns(nd.array): of model prediction agreements'''
    number_models = len(list_to_checkpoints)
    matrix = np.zeros((number_models,number_models))

    list_models = [load_model_from_checkpoint(list_to_checkpoints[i],model_type,dataset_name) 
        for i in range(len(list_to_checkpoints))]
    
    list_predictions = [get_prediction_on_data(list_models[i],dataloader,0,task,only_on_right_pred) 
        for i in range(len(list_models))]

    for i in range(number_models):
        for j in range(i+1):
            value = prediction_disagreement(list_predictions[i],list_predictions[j],task,only_on_right_pred)
            matrix[i,j] = value 
            matrix[j,i] = value
    return matrix

def get_tSNE_plot(list_to_models, model_type, dataset_name, dataloader, number_predictions, perplexity, task):
    '''computes the t_SNE plot like in the Loss Landscape Paper. Goes through every 
    checkpoint, uses the algorithm from the Loss Landscape paper to map it to 2d.
    Assumes every model has the same number of checkpoints.
    Args:
        list_to_models(list(str)): A list to paths of checkpoints of the models
        model_type(str): The type of the model
        dataset_name(str): The name of the dataset; together with model type identifies the model
        dataloader(torch.dataloader): The dataloader to get the samples to predict on
        number_predictions(int): The number of predictions to use for the plot
        perplexity(int): a parameter for the t-SNE method. between 5 und 50, gives how many 
            point should be considered for neighbourhood.
        task(str): either 'class' or 'seg', which task we are on
    Returns(list(tupels)): A List with every sublist being the 2d projection of a checkpoint 
        of the model
    '''
    #get a list of predictions, ever sublist for one model 
    number_models = len(list_to_models)
    all_predictions = [[] for _ in range(number_models)]
    for i,model_path in enumerate(list_to_models):
        checkpoint_paths = [path for path in os.listdir(model_path) if path.startswith('checkpoint')]
        checkpoint_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        for c_path in checkpoint_paths:
            model = load_model_from_checkpoint(os.path.join(model_path,c_path),model_type,dataset_name)
            all_predictions[i].append(get_prediction_on_data(model,dataloader,number_predictions, task))
    
    tsne_data_list = [c_pred for model_list in all_predictions for c_pred in model_list]
    tsne_data = torch.stack(tsne_data_list)

    tsne_data_trans = TSNE(perplexity=perplexity).fit_transform(tsne_data)

    # get the checkpoints of one model into one list ASSUMES EVERY MODEL HAS SAME NUMBER OF CHECKPOINTS
    length_tsne_trans = len(tsne_data_trans)
    pred_per_model = int(length_tsne_trans/number_models)
    tsne_per_model = [[] for _ in range(number_models)]
    for i in range(number_models):
        tsne_per_model[i]=tsne_data_trans[pred_per_model*i:pred_per_model*(i+1)]

    return tsne_per_model


    
