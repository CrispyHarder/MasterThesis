import matplotlib.pyplot as plt 
import seaborn as sns 
import math
import numpy as np

def plot_cosine_similarity(cos_sim_matrix,show=False,title='',xlabel='',ylabel='',save_path=''):
    '''plots the cosine similariy matrix of a number of models
    or model configurations'''
    n = np.shape(np.array(cos_sim_matrix))[0]
    ticks = math.floor(n/4)
    sns.set_theme()
    ax = sns.heatmap(cos_sim_matrix,xticklabels=ticks,yticklabels=ticks)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
