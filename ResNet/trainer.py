import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

from multistep_multigamma_lr import MultistepMultiGammaLR

from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

default_data_storage = os.path.join('..','storage','data')
default_save_dir = os.path.join('..','storage','models','ResNet')

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--data_storage', default=default_data_storage)
parser.add_argument('-device',default="0")
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default=default_save_dir, type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--nr_runs', type=int, default=100)
parser.add_argument('--runs_start_at',type=int,default=0)
parser.add_argument('--improvement_margin', type=float, default=0.5)
parser.add_argument('--breaking_condition', type=int, default=15, 
                    help='''After how many epochs without aggregated improvement 
                    of at least improvement_margin the training shall be stopped''')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    #set device
    print('Using GPU cuda {}'.format(args.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    #get the number of runs and number of already done runs  
    nr_runs = args.nr_runs
    runs_start_at = args.runs_start_at

    #run nr_runs often and save the models in the specified place
    for nr_run in range(runs_start_at,runs_start_at + nr_runs):

        # Check if the save_dir for the run exists or not,
        # path is save_dir/model_name(like resnet20)/run_ix
        save_dir_run = os.path.join(args.save_dir,args.arch,'run_{}'.format(nr_run))
        if not os.path.exists(save_dir_run):
            os.makedirs(save_dir_run)

        #add a writer to log training results for tensorboard
        writer = SummaryWriter(save_dir_run)

        # set the model, load and send to device
        model = resnet.__dict__[args.arch]()
        model.cuda()

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.evaluate, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        #dont know what this is doing
        cudnn.benchmark = True
        
        # params of original code are: 
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # params of moritz are: 
        # Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                        std=(0.2023, 0.1994, 0.2010))

        # get the directory to store the data
        data_storage = args.data_storage

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_storage, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_storage, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        # define loss function (criterion) and optimizer and lr_scheduler
        criterion = nn.CrossEntropyLoss().cuda()

        if args.half:
            model.half()
            criterion.half()

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        #use own lr_scheduler
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
        lr_scheduler = torch.optim.lr_scheduler.MultistepMultiGammaLR(optimizer,
                            milestones=[100, 150], gamma=[0.5,0.2],last_epoch=args.start_epoch - 1)

        if args.arch in ['resnet1202', 'resnet110']:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr*0.1


        if args.evaluate:
            _ , _ = validate(val_loader, model, criterion)
            return

        #preperation in order to get the number of epochs without improvement
        epochs_wo_improvement = 0
        improvement_margin = args.improvement_margin
        breaking_condition = args.breaking_condition

        for epoch in range(args.start_epoch, args.epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            prec1_train, loss_train = train(train_loader, model, criterion, optimizer, epoch)
            lr_scheduler.step()

            # evaluate on validation set
            prec1_val, loss_val = validate(val_loader, model, criterion)
            
            # log accuracy and loss for tensorboard
            writer.add_scalar('train/loss',loss_train,epoch)
            writer.add_scalar('train/accuracy',prec1_train,epoch)
            writer.add_scalar('val/loss',loss_val,epoch)
            writer.add_scalar('val/accuracy',prec1_val,epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1_val > best_prec1 + improvement_margin
            if is_best:
                best_prec1 = max(prec1_val, best_prec1)
                epochs_wo_improvement = 0
            else:
                epochs_wo_improvement += 1 

            #if early breaking condition is met and we are after last lr change, we end the training
            if epochs_wo_improvement >= breaking_condition and epoch > 150 + breaking_condition:
                break

            if epoch > 0 and epoch % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best,is_checkpoint = True, filename=os.path.join(save_dir_run, 'checkpoint.th'))

            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, is_checkpoint = False, filename=os.path.join(save_dir_run, 'model.th'))

        # empty the cache of the writer
        writer.flush()


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    
    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, is_checkpoint=True, filename='checkpoint.pth.tar'):
    """
    Save the training model if it has the best val performance
    """
    if is_checkpoint:
        torch.save(state, filename)
    else:
        if is_best:
            torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
