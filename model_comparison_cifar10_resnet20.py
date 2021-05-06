import os 
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from util.parameter_comparison import get_tSNE_plot, get_prediction_agreement_matrix, get_matrix_of_models, cosine_sim_model_params
from util.visualisation import plot_matrix_as_heatmap,plot_tSNE_scatter 

parser = argparse.ArgumentParser(description='Argparser to make experiments')
parser.add_argument('--device')
parser.add_argument('--runs_tsne',nargs='+',help='which runs to use for tsne plot')
parser.add_argument('--tsne_perplexity',default=50,type=int,help='perplexity parameter for tsne plot')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

list_to_models = os.path.join('storage','models','ResNet','cifar10','resnet20')

def main():
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
    ## get the t-SNE plots

    list_to_tsne_models = [os.path.join(list_to_models,run_nr) for run_nr in args.runs_tsne]
    tsne_list = get_tSNE_plot(list_to_tsne_models,'resnet20','cifar10',val_loader,5000,args.tsne_perplexity,'class')
    plot_tSNE_scatter(tsne_list,save_path=os.path.join('storage','results','t-SNE_cifar10_resnet20'))
    
if __name__ == '__main__':
    main()