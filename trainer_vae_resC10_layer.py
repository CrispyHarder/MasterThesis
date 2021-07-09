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

from models.parameter_learners.resnet_cifar10.layer.baseline import LayerVAEresC10, LayerCVAEresC10
from util.training.average_meter import AverageMeter
from util.saving import save_checkpoint, save_training_hparams, save_dict_values
from data.datasets.resnet_cifar10_dataset import Resnet_cifar10_layer_parameters_dataset


default_data_storage = os.path.join('storage', 'data', 'resnet_cifar10')
default_save_dir = os.path.join('storage','models','VAE','resnet_cifar10','layer')

parser = argparse.ArgumentParser(description='Layerwise VAE for Resnets trained on CIFAR10 in pytorch')

# the device to be used
parser.add_argument('-device', default="0")

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
parser.add_argument('--weight_kld', default=1.0 , type=float,
                    help='''the weight for the kld''')

# Model architecture
parser.add_argument('--arch', default='baseline', type=str,
                    help='The model to be used(dummy argument for now)')
parser.add_argument('--conditional', default=False, action='store_true',
                    help='whether the model should be conditional')
parser.add_argument('--in_channels', default=64, type=int,
                    help='''The number of channels of the inputs ''') 
parser.add_argument('--hidden_dims', default=[128,256,512], type=list,
                    help='''The dimensions of the hidden layers''')  
parser.add_argument('--latent_dim', default=64, type=int,
                    help='''The dimension of the latent space ''') 
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
    training_data = Resnet_cifar10_layer_parameters_dataset(path_to_data=args.data_storage, train=True)

    #for conditional version
    number_archs = training_data.number_archs
    number_layers = training_data.number_layers

    validation_data = Resnet_cifar10_layer_parameters_dataset(path_to_data=args.data_storage, train=False)

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
    for nr_run in range(runs_start_at, runs_start_at + nr_runs):

        # Check if the save_dir for the run exists or not,
        # path is save_dir/model_name(like resnet20)/run_ix
        if args.conditional:
            save_dir_run = os.path.join(args.save_dir, args.arch+'_conditional', 'run_{}'.format(nr_run))
        else:
            save_dir_run = os.path.join(args.save_dir, args.arch, 'run_{}'.format(nr_run))
        if not os.path.exists(save_dir_run):
            os.makedirs(save_dir_run)

        #add a writer to log training results for tensorboard
        writer = SummaryWriter(save_dir_run)

        # construct model and send it to GPU
        if args.conditional:
            model = LayerCVAEresC10(args.in_channels, args.latent_dim, args.hidden_dims, args.pre_interm_layers,
              args.interm_layers, args.sqrt_number_kernels, number_layers, number_archs).cuda()
        else:
            model = LayerVAEresC10(args.in_channels, args.latent_dim, args.hidden_dims, args.pre_interm_layers,
              args.interm_layers, args.sqrt_number_kernels).cuda()

        # configure optimizer    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)

        save_training_hparams(args, save_dir_run,
                                    {'number_archs':number_archs,
                                        'number_layers':number_layers})

        for epoch in range (args.start_epoch, args.start_epoch+args.epochs):
            best_loss = 100000

            # perform training for one epoch
            loss, recon_loss, kl_div = train_epoch(training_loader, model, optimizer, epoch)

            # compute on validation split
            val_loss, val_recon_loss, val_kl_div = validation(validation_loader, model)

            # log the scalar valuese
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/loss_recon',recon_loss, epoch)
            writer.add_scalar('train/loss_kl_div', kl_div, epoch)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/loss_recon', val_recon_loss, epoch)
            writer.add_scalar('val/loss_kl_div', val_kl_div, epoch)

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
        save_dict_values({'best_val_loss':best_loss,
                            'val_loss':val_loss,
                            'val_loss_recon':val_recon_loss},
                            save_dir_run)

        # empty the cache of the writer into the directory 
        writer.flush()
    print("Finished Training ")
    
def train_epoch(train_loader, model, optimizer, epoch):
    """
        Run one train epoch
    """
    global args
    # global train_data_variance
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    kl_divs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, mask, arch, layer) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        data = data.cuda()
        mask = mask.cuda()
        arch = arch.cuda()
        layer = layer.cuda()
        optimizer.zero_grad()
        
        data_recon, input, mu, log_var = model(data, arch=arch, layer=layer)
        loss_dict = model.loss_function(mask, data_recon, input, mu, log_var, M_N=args.weight_kld)
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']
        kl_div = loss_dict['KLD']
        loss.backward()

        optimizer.step()

        loss = loss.float()
        recon_loss = recon_loss.float()
        kl_div = kl_div.float()

        # update the loss dictionaries
        losses.update(loss.item(), data.size(0))
        recon_losses.update(recon_loss.item(), data.size(0))
        kl_divs.update(kl_div.item(), data.size(0))

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
    
    return losses.avg, recon_losses.avg, kl_divs.avg


def validation(val_loader,model):
    """
    Run evaluation
    """
    # global val_data_variance

    batch_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    kl_divs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (data, mask, arch, layer) in enumerate(val_loader):

            data = data.cuda()
            mask = mask.cuda()
            arch = arch.cuda()
            layer = layer.cuda()

            # compute output
            data_recon, input, mu, log_var = model(data, arch=arch, layer=layer)
            loss_dict = model.loss_function(mask, data_recon, input, mu, log_var, M_N=args.weight_kld)
            loss = loss_dict['loss']
            recon_loss = loss_dict['Reconstruction_Loss']
            kl_div = loss_dict['KLD']

            loss = loss.float()
            recon_loss = recon_loss.float()
            kl_div = kl_div.float()

            # update the loss dictionaries
            losses.update(loss.item(), data.size(0))
            recon_losses.update(recon_loss.item(), data.size(0))
            kl_divs.update(kl_div.item(), data.size(0))

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

    return losses.avg, recon_losses.avg, kl_divs.avg

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
