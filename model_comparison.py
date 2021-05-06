import os 
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from util.parameter_comparison import get_tSNE_plot, get_prediction_agreement_matrix, get_matrix_of_models, cosine_sim_model_params
from util.visualisation import plot_matrix_as_heatmap,plot_tSNE_scatter 

parser = argparse.ArgumentParser(description='Argparser to make experiments')
parser.add_argument('--device')

# specifying the model and dataset 
parser.add_argument('--model_type',default='resnet20')
parser.add_argument('--dataset_name',default='cifar10')
parser.add_argument('--task',choices=['class','seg'],default='class')

# which experiments to conduct 
parser.add_argument('--tsne', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--pred_agree_models', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--pred_agree_checkpoints', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--cosine_sim_models', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--cosine_sim_checkpoints', default=False, action='store_true', help='whether tsne plot shall be computed')

# tsne specific arguments 
parser.add_argument('--tsne_runs',nargs='+',help='which runs to use for tsne plot')
parser.add_argument('--tsne_perplexity',default=50,type=int,help='perplexity parameter for tsne plot')
parser.add_argument('--tsne_number_pred',default=5000,type=int,help='How many predictions are used for the tsne plot')

# pred_agree_models specific arguments, only works on older runs, where only last checkpoint was saved
parser.add_argument('--pam_start_runs',type=int,default=150,help='together with end runs determines, which runs to use')
parser.add_argument('--pam_end_runs',type=int,default=200)

# pred_agree_checkpoints specific arguments
parser.add_argument('--pac_run',type=int,default=215,help='together with end runs determines, which runs to use')

# cosine_sim_models specific arguments, only works on older runs, where only last checkpoint was saved
parser.add_argument('--csm_start_runs',type=int,default=150,help='together with end runs determines, which runs to use')
parser.add_argument('--csm_end_runs',type=int,default=200)

# cosine_sim_checkpoints specific arguments
parser.add_argument('--csc_run',type=int,default=215,help='together with end runs determines, which runs to use')

def main():

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # path to the models 
    list_to_models = os.path.join('storage','models','ResNet',args.dataset_name,args.model_type)

    #results path is where the plots are saved
    results_path = os.path.join('storage','results','ResNet',args.dataset_name,args.model_type)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if args.dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                            std=(0.2023, 0.1994, 0.2010))
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=os.path.join('storage','data'), train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=128, shuffle=False,
            num_workers=4, pin_memory=True)

    if args.tsne:
        print('computing tsne plot')
        ## get the t-SNE plots
        list_to_tsne_models = [os.path.join(list_to_models,'run_{}'.format(run_nr)) for run_nr in args.tsne_runs]
        tsne_list = get_tSNE_plot(list_to_tsne_models,args.model_type,args.dataset_name,val_loader,args.tsne_number_pred,args.tsne_perplexity,args.task)
        plot_tSNE_scatter(tsne_list,save_path=os.path.join(results_path,'t-SNE runs {}'.format(args.runs_tsne)))

    if args.pred_agree_models:
        print('computing pred_agree_models')
        ## operates on old resnet models, where no models, only checkpoints where saved
        list_to_checkpoints= [os.path.join(list_to_models,'run_{}'.format(run_nr),'checkpoint.th') 
            for run_nr in range(args.pam_start_runs,args.pam_end_runs)]
        pred_agree_matrix = get_prediction_agreement_matrix(list_to_checkpoints,args.model_type,args.dataset_name,val_loader,args.task)
        plot_matrix_as_heatmap(pred_agree_matrix,False,'Pred Agreement of models {} to {} '.format(args.pam_start_runs,args.pam_end_runs),
            xlabel='independent model', ylabel='independent model',
            save_path=os.path.join(results_path,'pred_agree_models'))
    
    if args.pred_agree_checkpoints:
        print('computing pred_agree_checkpoints')
        path_to_run = os.path.join(list_to_models,'run_'+args.pac_run)
        list_to_checkpoints= [os.path.join(path_to_run,c_point) 
            for c_point in os.listdir(path_to_run) if c_point.startswith('checkpoint')]
        pred_agree_matrix = get_prediction_agreement_matrix(list_to_checkpoints,args.model_type,args.dataset_name,val_loader,args.task)
        plot_matrix_as_heatmap(pred_agree_matrix,False,'Pred Agreement of model {} checkpoints'.format(args.pac_run),
            xlabel='checkpoints', ylabel='checkpoints',
            save_path=os.path.join(results_path,'pred_agree_cpoints_run_{} '.format(args.pac_run)))

    if args.cosine_sim_models:
        print('computing cosine_sim_models')
        list_to_checkpoints= [os.path.join(list_to_models,'run_{}'.format(run_nr),'checkpoint.th') 
            for run_nr in range(args.csm_start_runs,args.csm_end_runs)]
        cos_matrix = get_matrix_of_models(list_to_checkpoints,args.model_type,args.dataset_name,cosine_sim_model_params)
        plot_matrix_as_heatmap(cos_matrix,False,'Cosine Similarity of models {} to {} '.format(args.csm_start_runs,args.csm_end_runs),
            xlabel='independent model', ylabel='independent model',
            save_path=os.path.join(results_path,'cosine_similarity_models'))
    
    if args.cosine_sim_checkpoints:
        print('computing cosine_sim_checkpoints')
        path_to_run = os.path.join(list_to_models,'run_'+args.csc_run)
        list_to_checkpoints= [os.path.join(path_to_run,c_point) 
            for c_point in os.listdir(path_to_run) if c_point.startswith('checkpoint')]
        cos_matrix = get_matrix_of_models(list_to_checkpoints,args.model_type,args.dataset_name,cosine_sim_model_params)
        plot_matrix_as_heatmap(cos_matrix,False,'Cosine Similarity of checkpoints of model {} '.format(args.csc_run),
            xlabel='checkpoints', ylabel='checkpoints',
            save_path=os.path.join(results_path,'cosine_similarity_c_points_model_{}'.format(args.csc_run)))
        
if __name__ == '__main__':
    main()