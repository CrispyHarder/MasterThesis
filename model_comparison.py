import os 
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import re 
from util.parameter_comparison import get_tSNE_plot, get_prediction_disagreement_matrix, get_matrix_of_models, cosine_sim_model_params
from util.visualisation import plot_matrix_as_heatmap,plot_tSNE_scatter 

parser = argparse.ArgumentParser(description='Argparser to make experiments')
parser.add_argument('--device')

# specifying the model and dataset 
parser.add_argument('--model_type',default='resnet20')
parser.add_argument('--dataset_name',default='cifar10')
parser.add_argument('--task',choices=['class','seg'],default='class')

# which experiments to conduct 
parser.add_argument('--tsne', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--pred_disagree_models', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--pred_disagree_checkpoints', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--cosine_sim_models', default=False, action='store_true', help='whether tsne plot shall be computed')
parser.add_argument('--cosine_sim_checkpoints', default=False, action='store_true', help='whether tsne plot shall be computed')

# tsne specific arguments 
parser.add_argument('--tsne_runs',nargs='+',help='which runs to use for tsne plot')
parser.add_argument('--tsne_perplexity',default=50,type=int,help='perplexity parameter for tsne plot')
parser.add_argument('--tsne_number_pred',default=5000,type=int,help='How many predictions are used for the tsne plot')

# pred_disagree_models specific arguments, only works on older runs, where only last checkpoint was saved
parser.add_argument('--pdm_start_runs',type=int,default=150,help='together with end runs determines, which runs to use')
parser.add_argument('--pdm_end_runs',type=int,default=200)

# pred_disagree_checkpoints specific arguments
parser.add_argument('--pdc_run',type=int,default=208,help='together with end runs determines, which runs to use')

# cosine_sim_models specific arguments, only works on older runs, where only last checkpoint was saved
parser.add_argument('--csm_start_runs',type=int,default=150,help='together with end runs determines, which runs to use')
parser.add_argument('--csm_end_runs',type=int,default=200)

# cosine_sim_checkpoints specific arguments
parser.add_argument('--csc_run',type=int,default=215,help='together with end runs determines, which runs to use')

# parameters for the dataloader
parser.add_argument('--num_workers', default=4, type=int, help='How many workers for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='batch size for the dataloader')

def main():

    args = parser.parse_args()

    # path to the models 
    list_to_models = os.path.join('storage','models','ResNet',args.dataset_name,args.model_type)

    #results path is where the plots are saved
    results_path = os.path.join('storage','results','ResNet',args.dataset_name,args.model_type)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Parameters for the Dataloader on cpu
    batch_size = 8
    num_worker = 0

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        batch_size = args.batch_size
        num_worker = args.num_workers

    if args.dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                            std=(0.2023, 0.1994, 0.2010))
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=os.path.join('storage','data'), train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_worker, pin_memory=True)

    if args.tsne:
        print('computing tsne plot')
        ## get the t-SNE plots
        list_to_tsne_models = [os.path.join(list_to_models,'run_{}'.format(run_nr)) for run_nr in args.tsne_runs]
        tsne_list = get_tSNE_plot(list_to_tsne_models,args.model_type,args.dataset_name,val_loader,args.tsne_number_pred,args.tsne_perplexity,args.task)
        plot_tSNE_scatter(tsne_list,save_path=os.path.join(results_path,'t-SNE runs {}'.format(args.runs_tsne)))

    if args.pred_disagree_models:
        print('computing pred_disagree_models')
        ## operates on old resnet models, where no models, only checkpoints where saved
        list_to_checkpoints= [os.path.join(list_to_models,'run_{}'.format(run_nr),'checkpoint.th') 
            for run_nr in range(args.pdm_start_runs,args.pdm_end_runs)]
        pred_agree_matrix = get_prediction_disagreement_matrix(list_to_checkpoints,args.model_type,args.dataset_name,val_loader,args.task)
        plot_matrix_as_heatmap(pred_agree_matrix,False,'Pred Disagreement of models {} to {} '.format(args.pdm_start_runs,args.pdm_end_runs),
            xlabel='independent model', ylabel='independent model',
            save_path=os.path.join(results_path,'pred_disagree_models'))
    
    if args.pred_disagree_checkpoints:
        print('computing pred_disagree_checkpoints')
        path_to_run = os.path.join(list_to_models,'run_'+str(args.pdc_run))
        list_to_checkpoints= [os.path.join(path_to_run,c_point) 
            for c_point in os.listdir(path_to_run) if c_point.startswith('checkpoint')]
        list_to_checkpoints.sort(key=lambda f: int(re.sub('\D', '', f)))
        pred_agree_matrix = get_prediction_disagreement_matrix(list_to_checkpoints,args.model_type,args.dataset_name,val_loader,args.task)
        plot_matrix_as_heatmap(pred_agree_matrix,False,'Pred Disagreement of model {} checkpoints'.format(args.pdc_run),
            xlabel='checkpoints', ylabel='checkpoints',
            save_path=os.path.join(results_path,'pred_disagree_cpoints_run_{} '.format(args.pdc_run)))

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
        path_to_run = os.path.join(list_to_models,'run_'+str(args.csc_run))
        list_to_checkpoints= [os.path.join(path_to_run,c_point) 
            for c_point in os.listdir(path_to_run) if c_point.startswith('checkpoint')]
        list_to_checkpoints.sort(key=lambda f: int(re.sub('\D', '', f)))
        cos_matrix = get_matrix_of_models(list_to_checkpoints,args.model_type,args.dataset_name,cosine_sim_model_params)
        plot_matrix_as_heatmap(cos_matrix,False,'Cosine Similarity of checkpoints of model {} '.format(args.csc_run),
            xlabel='checkpoints', ylabel='checkpoints',
            save_path=os.path.join(results_path,'cosine_similarity_c_points_model_{}'.format(args.csc_run)))
        
if __name__ == '__main__':
    main()