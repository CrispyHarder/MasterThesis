from __future__ import print_function

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from VQ_VAE.basic_gc.vq_vae import Model
from util.average_meter import AverageMeter
from util.saving import save_checkpoint


default_data_storage = os.path.join('storage','data')
default_save_dir = os.path.join('storage','models','VQVAE')

parser = argparse.ArgumentParser(description='VQ-VAE for CIFAR10 in pytorch')

# the device to be used
parser.add_argument('-device',default="0")

# training specifics 
parser.add_argument('--nr_runs', type=int, default=100)
parser.add_argument('--runs_start_at',type=int,default=0)
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')

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
parser.add_argument('--arch', default='basic_model', type=str,
                    help='The model to be used(dummy argument for now)')
parser.add_argument('--num_hiddens', default=128, type=int,
                    help='''The dimension of the hidden layers
                     of the convolution in encoder and decoder ''') 
parser.add_argument('--num_residual_hiddens', default=32, type=int,
                    help='''The dimension of the hidden layers 
                    of the residual stack''')  
parser.add_argument('--num_residual_layers', default=2, type=int,
                    help='''The number of residual layers in
                    encoder and decoder''') 
parser.add_argument('--embedding_dim', default=64, type=int,
                    help='''The dimension of the embedding(codebook)
                    vectors''')
parser.add_argument('--num_embeddings', default=512, type=int,
                    help='''The number of embedding(codebook) vectors''')           


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
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

#breaking conditions 
parser.add_argument('--improvement_margin', type=float, default=0.5)
parser.add_argument('--breaking_condition', type=int, default=15, 
                    help='''After how many epochs without aggregated improvement 
                    of at least improvement_margin the training shall be stopped''')



def main():
    global args
    args = parser.parse_args()

    #set device
    print('Using GPU cuda {}'.format(args.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    #get the number of runs and number of already done runs  
    nr_runs = args.nr_runs
    runs_start_at = args.runs_start_at

    # load train data 
    # Moritz values for normaisation: mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010)
    # normalisation here is only to scale images to [-0.5,0.5] 
    training_data = datasets.CIFAR10(root=args.data_storage, train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,0.5,0.5),
                                            std=(1,1,1))
                                    ]))

    validation_data = datasets.CIFAR10(root=args.data_storage, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,0.5,0.5),
                                            std=(1,1,1))
                                    ]))

    # put train data into DataLoader
    training_loader = DataLoader(training_data, 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)

    validation_loader = DataLoader(validation_data,
                                batch_size=32,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)

    #to use in loss 
    global train_data_variance
    global val_data_variance
    train_data_variance = np.var(training_data.data / 255.0)
    val_data_variance = np.var(validation_data.data / 255.0)


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
        model = Model(args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens,
              args.num_embeddings, args.embedding_dim, 
              args.commitment_cost, args.decay).cuda()

        # configure optimizer    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)

        for epoch in range (args.start_epoch, args.start_epoch+args.epochs):
            
            # perform training for one epoch
            loss, recon_loss, vq_loss, perplexity = train_epoch(training_loader,model,optimizer, epoch)

            # compute on validation split
            val_loss, val_recon_loss, val_vq_loss, val_perplexity, data_input, data_recon = validation(validation_loader, model)

            # log the scalar valuese
            writer.add_scalar('train/loss',loss,epoch)
            writer.add_scalar('train/loss_recon',recon_loss,epoch)
            writer.add_scalar('train/loss_vq',vq_loss,epoch)
            writer.add_scalar('train/perplexity',perplexity,epoch)

            writer.add_scalar('val/loss',val_loss,epoch)
            writer.add_scalar('val/loss_recon',val_recon_loss,epoch)
            writer.add_scalar('val/loss_vq',val_vq_loss,epoch)
            writer.add_scalar('val/perplexity',val_perplexity,epoch)

            # log the reconstructed images (add 0.5 to get image into right range)
            writer.add_images('val/img_recon', data_recon+0.5, epoch)
            writer.add_images('val/img_orig', data_input+0.5, epoch)

            if epoch > 0 and epoch % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict()
                }, is_checkpoint = True, filename=os.path.join(save_dir_run, 'checkpoint.th'))

            save_checkpoint({
                'state_dict': model.state_dict()
            }, is_checkpoint = False, filename=os.path.join(save_dir_run, 'model.th'))
        
        # save the hyperparams using the writer
        writer.add_hparams({'batch_size':args.batch_size,
                            'lr':args.learning_rate,
                            'commitment_cost':args.commitment_cost,
                            'decay':args.decay,
                            'model':args.arch,
                            'num_hiddens':args.num_hiddens,
                            'num_residual_hiddens':args.num_residual_hiddens,
                            'num_residual_layers':args.num_residual_layers,
                            'embedding_dim':args.embedding_dim,
                            'num_embeddings':args.num_embeddings,
                            'nr_run':nr_run},
                            {'val_loss':val_loss,
                            'va_loss_recon':val_recon_loss,
                            'val_loss_vq':val_vq_loss,
                            'end_perplexity':val_perplexity})
        # empty the cache of the writer into the directory 
        writer.flush()

def train_epoch(train_loader,model, optimizer, epoch):
    """
        Run one train epoch
    """
    global train_data_variance

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    vq_losses = AverageMeter()
    perplexities = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        data = data.cuda()
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_loss = F.mse_loss(data_recon, data)/train_data_variance
        loss = recon_loss + vq_loss
        loss.backward()

        optimizer.step()

        loss = loss.float()
        recon_loss = recon_loss.float()
        vq_loss = vq_loss.float()
        perplexity = perplexity.float()

        # update the loss dictionaries
        losses.update(loss.item(), data.size(0))
        recon_losses.update(recon_loss.item(), data.size(0))
        vq_losses.update(vq_loss.item(), data.size(0))
        perplexities.update(perplexity.item(),data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Perplexity {perple.val:.3f} ({perple.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, perple=perplexities))
    
    return losses.avg, recon_losses.avg, vq_losses.avg, perplexities.avg


def validation(val_loader,model):
    """
    Run evaluation
    """
    global val_data_variance

    batch_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    vq_losses = AverageMeter()
    perplexities = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):

            data = data.cuda()

            # compute output
            vq_loss, data_recon, perplexity = model(data)
            recon_loss = F.mse_loss(data_recon, data)/val_data_variance
            loss = recon_loss + vq_loss

            loss = loss.float()
            recon_loss = recon_loss.float()
            vq_loss = vq_loss.float()
            perplexity = perplexity.float()

            # update the loss dictionaries
            losses.update(loss.item(), data.size(0))
            recon_losses.update(recon_loss.item(), data.size(0))
            vq_losses.update(vq_loss.item(), data.size(0))
            perplexities.update(perplexity.item(),data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Perplexity {perple.val:.3f} ({perple.avg:.3f})\t'.format( i, 
                        len(val_loader), batch_time=batch_time,
                      loss=losses, perple=perplexities))

            # take the images of the first validation batch and also return them 
            # for visual inspection
            if i == 0:
                data_input = data.data
                data_recon = data_recon.data
                data_recon_return = data_recon

    print(' * Loss {loss.avg:.3f}'
          .format(loss=losses))

    return losses.avg, recon_losses.avg, vq_losses.avg, perplexities.avg, data_input, data_recon_return 

if __name__ == '__main__':
    main()
