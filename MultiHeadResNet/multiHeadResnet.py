'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

#__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

#HELPER
def combineMembers(outs):
    if type(outs) == list or type(outs) == tuple:
        tmp = torch.stack(outs,0)
    else:
        tmp = outs

    return torch.sum(F.softmax(tmp,2),0)/len(outs)


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self,x):
        return F.avg_pool2d(x, x.size()[3])


class MultiHeadResNet(nn.Module):
    def __init__(self, num_blocks, num_heads, num_classes=10, split_lvl=3):
        super(MultiHeadResNet, self).__init__()
        print("Num Heads", num_heads)
        self.headblocks = nn.ModuleList()
        self.headlinears = nn.ModuleList()
        self.split_lvl = split_lvl

        #Really messy implementation to ensure backwards compatibility to older runs
        head_block_lists = [[] for _ in range(num_heads)]

        #Implemented split levels
        if split_lvl not in  [0, 1, 2, 3, 4]:
            print("Split_lvl {}".format(split_lvl) + " is not implemented.")
            raise NotImplementedError

        block = BasicBlock

        self.in_planes = 16
        self.num_heads = num_heads
        self.num_classes = num_classes

        if split_lvl > 0:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
        else:
            inplane_c = self.in_planes
            for nh in range(num_heads):
                self.in_planes = inplane_c
                head_block_lists[nh].append(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False))
                head_block_lists[nh].append(nn.BatchNorm2d(16))
                head_block_lists[nh].append(nn.ReLU(inplace=False))

        if split_lvl > 1:
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        else:
            inplane_c = self.in_planes
            for nh in range(num_heads):
                self.in_planes = inplane_c
                head_block_lists[nh].append(self._make_layer(block, 16, num_blocks[0], stride=1))

        if split_lvl > 2:
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        else:
            inplane_c = self.in_planes
            for nh in range(num_heads):
                self.in_planes = inplane_c
                head_block_lists[nh].append(self._make_layer(block, 32, num_blocks[1], stride=2))

        #self.heads = nn.ModuleList()

        if split_lvl > 3:
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        else:
            inplane_c = self.in_planes
            for nh in range(num_heads):
                self.in_planes = inplane_c
                head_block_lists[nh].append(self._make_layer(block, 64, num_blocks[2], stride=2))

        inplane_c = self.in_planes
        for _ in range(num_heads):
            self.in_planes = inplane_c
            self.headlinears.append(nn.Conv2d(64, num_classes, kernel_size=1))

        for layers in head_block_lists:

            if len(layers) == 1:
                self.headblocks.append(layers[0])
            elif len(layers) > 1:
                self.headblocks.append(nn.Sequential(*layers))
            else:
                continue

        #for _ in range(num_heads):
        #    self.in_planes = inplane_c
        #    self.headblocks.append(self._make_layer(block, 64, num_blocks[2], stride=2))
        #    self.headlinears.append(nn.Conv2d(64, num_classes, kernel_size=1))
            #head = nn.Sequential()
            #head.add_module("head_block", self._make_layer(block, 64, num_blocks[2], stride=2))
            #head.add_module("head_avgpool",  GlobalAvgPool())
            #head.add_module("head_outputconv", nn.Conv2d(64, num_classes, kernel_size=1))

            #self.heads.append(head)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        outs = []
        out = x

        if self.split_lvl > 0:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            #TODO Check wegen Relu. Hier ist ein Problem wegen dem Relu....
            for i in range(self.num_heads):
                out_c = out.clone()
                out_c = self.headblocks[i](out_c)
                out_c = F.avg_pool2d(out_c, out_c.size()[3])
                out_c = self.headlinears[i](out_c)
                outs.append(out_c)
            return outs

        if self.split_lvl > 1:
            out = self.layer1(out)
        else:
            for i in range(self.num_heads):
                out_c = out.clone()
                out_c = self.headblocks[i](out_c)
                out_c = F.avg_pool2d(out_c, out_c.size()[3])
                out_c = self.headlinears[i](out_c)
                outs.append(out_c)
            return outs

        if self.split_lvl > 2:
            out = self.layer2(out)
        else:
            for i in range(self.num_heads):
                out_c = out.clone()
                out_c = self.headblocks[i](out_c)
                out_c = F.avg_pool2d(out_c, out_c.size()[3])
                out_c = self.headlinears[i](out_c)
                outs.append(out_c)
            return outs

        if self.split_lvl > 3:
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
        else:
            for i in range(self.num_heads):
                out_c = out.clone()
                out_c = self.headblocks[i](out_c)
                out_c = F.avg_pool2d(out_c, out_c.size()[3])
                out_c = self.headlinears[i](out_c)
                outs.append(out_c)
            return outs

        for i in range(self.num_heads):
            out_c = out.clone()
            out_c = self.headlinears[i](out_c)
            outs.append(out_c)
        return outs

'''
def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])
'''



'''
def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
'''


class MHEnsembleLoss(torch.nn.Module):
    def __init__(self, num_heads, criterion, LAMBDA=0., PHI=0.):
        super().__init__()
        self.num_heads = num_heads
        self.criterion = criterion
        self.criterion2 = nn.NLLLoss()
        self.LAMBDA = LAMBDA
        self.PHI = PHI

    def forward(self,outputs, output, target_var):
        # output = output.float()
        # i_rad = -1 # measure_IRad(output)
        # Energy = 0  # energy_loss_term(output, kappa = 1e-4) #kappa = 1 on record --> kappa = 1e-4. TRY NEXT: kappa = 1e-5
        # E = 0 --> Basic multihead training
        # print(target_var.size())
        # target = torch.zeros(output.size())
        # for i in range(output.size()[0]):
        #     target[i][target_var[i]] = 1
        # target = target.cuda()
        # target_var = target
        # if model.num_heads == -1:
        #     # print(output.size(),target_var.size())
        #     output = F.softmax(output, dim=1)  # convert to probabilities | [batch_size, num_classes]
        #
        #     pred = output.argmax(dim=1).float()  # give predictions based on maximum probability [batch_size]
        #
        #     loss = criterion(output, target_var)
        #     loss_heads = torch.zeros(1)
        #     return loss, loss_heads, pred
        # print(output.size())
        # ens_out = output  # majority vote across heads | [batch_size, num_classes]
        # outputs = F.softmax(outputs, dim=2)  # convert to probabilities | "[num_heads, batch_size, num_classes]"

        # pred_ens = ens_out.argmax(dim=1).float()
        # print (target_var)
        # print(ens_out.size(),target_var.size())
        # print(output.size(),target_var.size())
        # print(target_var.size())
        loss_Ens = self.criterion2(torch.log(output), target_var)#self.criterion(output, target_var)
        loss_heads = torch.zeros(self.num_heads)

        # uncert = -torch.sum((ens_out) * (torch.log((ens_out + 1e-18))).cuda(), dim=1) / torch.log(
        #     (torch.Tensor([model.num_classes])).cuda())

        for i in range(self.num_heads):  # change to num_heads-1 for uncertainty
            # target_i = target_var.clone()#.detach().clone()
            # if False:
            #     target_i[uncert < float(i) / model.num_heads] = -1  # IGNORE_INDEX
            #     target_i[uncert >= float(i + 1) / model.num_heads] = -1  # IGNORE_INDEX
            #
            #     # print("Head", i, criterion(F.softmax(output[i],dim=1), target_i))
            # pred_i = output[i].argmax(dim=1).float()
            loss_heads[i] = self.criterion(outputs[i], target_var)

        loss_Heads = loss_heads.sum() / self.num_heads
        loss = (1 - self.LAMBDA) * loss_Heads + self.LAMBDA * loss_Ens

        # loss = (1-PHI) * loss + PHI * Energy

        return loss, loss_Ens, loss_heads, torch.zeros(1)#Energy


def train_epoch(net, opt, loss, train_loader, step, writer, lr_scheduler, device):

    net.train()

    correct_train = 0
    num_train = 0

    cum_loss = 0.

    net.train()
    for batch in tqdm(train_loader):
        opt.zero_grad()
        img, mask = batch

        img = img.to(device)
        mask = mask.to(device)
        outs = net(img)


        for i in range(len(outs)):
            outs[i] = outs[i].view(outs[i].size()[0:2])

        out = combineMembers(outs)


        preds = torch.argmax(out, 1)
        # l, overall_l , indi_l, kl_loss = loss(outs, out, mask)
        # print(len(outs), outs[0].size(),out.size())
        l, overall_l , indi_l, kl_loss = loss(outs, out, mask)
        l.backward()
        opt.step()

        correct_train += torch.sum(preds == mask).item()
        num_train += mask.size(0)
        cum_loss += l
        # print("Loss " + str(overall_l.item()) + " KL " + str(kl_loss.item()), end="")
    try:
        writer.add_scalar('train/ACC', correct_train / num_train, step)
        writer.add_scalar('train/Loss', cum_loss, step)
    except:
        pass

    print('Train ACC: %.3f' % (correct_train / num_train))
    print('Train Loss: %.3f'% cum_loss," epoch: ", step)


if __name__ == "__main__":

    '''
    bla = torch.stack([torch.linspace(0, 1, 11),torch.linspace(0, 1, 11)])
    bla = torch.stack([bla,bla,bla])
    take_values = torch.tensor([2,3])
    tv_map = F.one_hot(take_values,num_classes=11).bool()
    print(tv_map)

    print(bla[:,torch.logical_not( tv_map)].view(3,2,10))
    '''


    from torchvision.datasets import CIFAR10, CIFAR100
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # try:
    #     from Losses import MulHCELossCosDistDet
    # except:
    #     from .Losses import MulHCELossCosDistDet
    from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, \
        RandomVerticalFlip, RandomCrop
    from torch.utils.data import DataLoader, Dataset
    import torchvision

    train_loader = DataLoader(CIFAR10(os.path.join('gris','gris-f','homestud','charder','MasterThesis','storage','data','CIFAR10'), 
            train=True, transform=Compose([RandomHorizontalFlip(), RandomCrop(32,4), ToTensor(), 
            Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]),download=True), 
            batch_size=128, shuffle=True)
    net = MultiHeadResNet([3,3,3], 1).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=2e-4)
    # loss = MulHCELossCosDistDet(0.5, 2.0, 0.5).cuda()
    loss = MHEnsembleLoss(net.num_heads, nn.CrossEntropyLoss(), LAMBDA=0.5, PHI=0).cuda()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 160, 240, 320], gamma=0.1)

    softmax_layer = nn.Softmax2d()
    for epoch in range(320):
        train_epoch(net, opt, loss, train_loader, epoch, None, lr_scheduler, torch.device("cuda:0"))



