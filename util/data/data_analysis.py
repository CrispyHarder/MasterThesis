
def histogramm_ensemble_accuracy_per_point(list_to_models,dataloader,bin_steps,model_type,dataset_name,task):
    '''computes the predictions for all models given and then computes per data point, 
    how many ensembles members predict it right. Gives insight whether there are points, 
    that tend to be predicted right less times
    Args:
        list_to_models(list(str)): A list to paths to the models
        model_type(str): The type of the model
        dataset_name(str): The name of the dataset; together with model type identifies the model
        dataloader(torch.dataloader): The dataloader to get the samples to predict on
        task(str): either 'class' or 'seg', which task we are on 
    
    Returns(nd.array):a histogramm over how often points get predicted right by the ensemble members in percent
        for a model with 90 percent acc, some points might get predicted right by all members and some nearly never'''
    # get all model 
    # get all predictions
    # get ground truth 
    # compute number of right pred per data point
    # convert to percentage 
    # hist over these values 
    
    
