from __future__ import print_function

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models.parameter_learners.resnet_cifar10.baseline_models import LayerVQVAEresC10
from util.average_meter import AverageMeter
from util.saving import save_checkpoint
from data.datasets.resnet_cifar10_dataset import Resnet_cifar10_layer_parameters_dataset


default_data_storage = os.path.join('storage', 'models', 'ResNet', 'cifar10')
default_save_dir = os.path.join('storage', 'models', 'VQVAE', 'resnet_cifar10', 'layer')

parser = argparse.ArgumentParser(description='Layerwise VQVAE for Resnets trained on CIFAR10 in pytorch')

# the device to be used
parser.add_argument('-device',default="0")

# training specifics 
parser.add_argument('--runs',type=int,default=1, help = 'number of runs')
parser.add_argument('--runs_start_at',type=int, default=0,
                    help='how many runs already have been done')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--not_verbose', default=False, action='store_true',
                    help='If switch is activated, no performance outputs are printed')
                    
# optimizer configuration/ loss function specifics
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--commitment_cost', default=0.25, type=float,
                    help='momentum')
parser.add_argument('--decay', default=0.99, type=float,
                    help='''decay (default: 0.99) for the moving ) 
                    average update of embeddings. If decay is used, 
                    the EMA vq model is used instead of the standart one 
                    from the paper''')

# Model architecture
parser.add_argument('--arch', default='baseline', type=str,
                    help='The model to be used(dummy argument for now)')
parser.add_argument('--in_channels', default=64, type=int,
                    help='''The number of channels of the inputs ''') 
parser.add_argument('--embedding_dim', default=32, type=int,
                    help='''The dimension of the embedding(codebook)
                    vectors''')
parser.add_argument('--num_embeddings', default=128, type=int,
                    help='''The number of embedding(codebook) vectors''')    
parser.add_argument('--hidden_dims', default=[256], type=int,
                    help='''The dimensions of the hidden layers''')  
parser.add_argument('--pre_interm_layers', default=1, type=int,
                    help='''The number of pre/post-intermediate layers to make 
                    the network deeper''')
parser.add_argument('--interm_layers', default=1, type=int,
                    help='''The number of intermediate layers to make 
                    the network deeper''') 
parser.add_argument('--sqrt_number_kernels', default=8, type=int,
                    help='''a parameter for the network depending on the number
                    of filters per layer''')          


# saving data,model states and results 
parser.add_argument('--data_storage', default=default_data_storage,
                    help='where the train/val/test data is saved')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default=default_save_dir, type=str)

# prints/outputs in console
parser.add_argument('--print-freq', '-p', default=400, type=int,
                    metavar='N', help='print frequency (default: 50)')


def main():
    global args
    args = parser.parse_args()

    #set device
    print('Using GPU cuda {}'.format(args.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    #get number of runs and at which index runs start 
    nr_runs = args.runs
    runs_start_at = args.runs_start_at

    # load train data 
    training_data = Resnet_cifar10_layer_parameters_dataset(path_to_data=args.data_storage,train=True)

    validation_data = Resnet_cifar10_layer_parameters_dataset(path_to_data=args.data_storage,train=False)

    # put train data into DataLoader
    training_loader = DataLoader(training_data, 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=False)

    validation_loader = DataLoader(validation_data,
                                batch_size=32,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=False)

    # #to use in loss 
    # global train_data_variance
    # global val_data_variance
    # train_data_variance = np.var(training_data.data / 255.0)
    # val_data_variance = np.var(validation_data.data / 255.0)

    #run nr_runs often and save the models in the specified place
    for nr_run in range(runs_start_at,runs_start_at + nr_runs):

        # Check if the save_dir for the run exists or not,
        # path is save_dir/model_name(like resnet20)/run_ix
        save_dir_run = os.path.join(args.save_dir,args.arch,'run_{}'.format(nr_run))
        if not os.path.exists(save_dir_run):
            os.makedirs(save_dir_run)

        #add a writer to log training results for tensorboard
        writer = SummaryWriter(save_dir_run)

        # construct model and send it to GPU
        model = LayerVQVAEresC10(args.in_channels, args.embedding_dim, args.num_embeddings, 
                args.commitment_cost, args.decay, args.hidden_dims, args.pre_interm_layers,
                args.interm_layers, args.sqrt_number_kernels).cuda()

        # configure optimizer    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)

        writer.add_hparams({'batch_size':args.batch_size,
                            'lr':args.learning_rate,
                            'commitment cost': args.commitment_cost,
                            'decay': args.decay, 
                            'model':args.arch,
                            'in_channels':args.in_channels,
                            'embedding dim': args.embedding_dim,
                            'num embeddings': args.num_embeddings,
                            #'hidden_dims':args.hidden_dims,
                            'pre_interm_layers':args.pre_interm_layers,
                            'interm_layers':args.interm_layers,
                            'nr_run':nr_run,
                            'sqrt_number_kernels':args.sqrt_number_kernels},
                            {'start time':time.time()})

        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            best_loss = 100000

            # perform training for one epoch
            loss, recon_loss, vq_loss, perplex = train_epoch(training_loader, model, optimizer, epoch)

            # compute on validation split
            val_loss, val_recon_loss, val_vq_loss, val_perplex = validation(validation_loader, model)

            # log the scalar valuese
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/loss_recon', recon_loss, epoch)
            writer.add_scalar('train/loss_vq_loss', vq_loss, epoch)
            writer.add_scalar('train/perplexity', perplex, epoch)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/loss_recon', val_recon_loss, epoch)
            writer.add_scalar('val/loss_vq_loss', val_vq_loss, epoch)
            writer.add_scalar('train/perplexity', val_perplex, epoch)

            if epoch > 0 and epoch % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict()
                }, is_checkpoint = True, filename=os.path.join(save_dir_run, 'checkpoint_{}.th'.format(epoch+1)))

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict()
                }, is_checkpoint = False, is_best = True, filename=os.path.join(save_dir_run, 'model.th'))

            print("Run nr {}, epoch {} finished training".format(nr_run,epoch), end="\r")

        # save the hyperparams using the writer
        writer.add_hparams({'last epoch':epoch},
                            {'best_val_loss':best_loss,
                            'val_loss':val_loss,
                            'val_loss_recon':val_recon_loss})

        # empty the cache of the writer into the directory 
        writer.flush()
    print("Finished Training ")
    
def train_epoch(train_loader, model, optimizer, epoch):
    """
        Run one train epoch
    """
    # global train_data_variance
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    vq_losses = AverageMeter()
    perplexities = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, mask, arch) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        data = data.cuda()
        optimizer.zero_grad()

        #get outputs
        vq_loss, data_recon, perplexity = model(data)
        loss_dict = model.loss_function(input, mask, data_recon, vq_loss)
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']

        #optimize
        loss.backward()
        optimizer.step()

        #get the values
        loss = loss.float()
        recon_loss = recon_loss.float()
        vq_loss = vq_loss.float()
        perplexity = perplexity.float()

        # update the loss dictionaries
        losses.update(loss.item(), data.size(0))
        recon_losses.update(recon_loss.item(), data.size(0))
        vq_losses.update(vq_loss.item(), data.size(0))
        perplexities.update(perplexity.item(), data.size())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and not args.not_verbose:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
    
    return losses.avg, recon_losses.avg, vq_losses.avg, perplexities.avg


def validation(val_loader, model):
    """
    Run evaluation
    """
    # global val_data_variance

    batch_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    vq_losses = AverageMeter()
    perplexities = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (data, mask, arch) in enumerate(val_loader):

            data = data.cuda()

            # compute output
            vq_loss, data_recon, perplexity = model(data)
            loss_dict = model.loss_function(input, mask, data_recon, vq_loss)
            loss = loss_dict['loss']
            recon_loss = loss_dict['Reconstruction_Loss']

            loss = loss.float()
            recon_loss = recon_loss.float()
            vq_loss = vq_loss.float()
            perplexity = perplexity.float()

            # update the loss dictionaries
            losses.update(loss.item(), data.size(0))
            recon_losses.update(recon_loss.item(), data.size(0))
            vq_losses.update(vq_loss.item(), data.size(0))
            perplexities.update(perplexity.item(), data.size())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and not args.not_verbose:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format( i, 
                        len(val_loader), batch_time=batch_time,
                      loss=losses))

    if not args.not_verbose:
        print(' * Loss {loss.avg:.3f}'
          .format(loss=losses))

    return losses.avg, recon_losses.avg, vq_losses.avg, perplexities.avg

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
