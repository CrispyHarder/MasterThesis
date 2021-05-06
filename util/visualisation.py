import matplotlib.pyplot as plt 
import seaborn as sns 
import math
import numpy as np

def plot_matrix_as_heatmap(matrix,show=False,title='',xlabel='',ylabel='',save_path=''):
    '''plots the cosine similariy matrix of a number of models
    or model configurations'''
    n = np.shape(np.array(matrix))[0]
    ticks = math.floor(n/4)
    sns.set_theme()
    ax = sns.heatmap(matrix,xticklabels=ticks,yticklabels=ticks,cmap='bwr')
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_tSNE_scatter(projections_per_model, show=False, save_path=''):
    '''plots the results of a t-SNE computation as in the LossLandscape Paper'''
    number_models = len(projections_per_model)
    for i in range(number_models):
        plt.scatter(projections_per_model[i][:,0],projections_per_model[i][:,1],label='model {}'.format(i))
    plt.title('t-SNE plots of models')
    plt.xlabel('t-SNE axis 1')
    plt.xlabel('t-SNE axis 2')
    plt.legend()
    plt.savefig(save_path)
    if show:
        plt.show()
