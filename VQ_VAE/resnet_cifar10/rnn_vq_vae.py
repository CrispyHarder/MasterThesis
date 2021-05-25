import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRnnEncoder(nn.Module):
    def __init__(self, in_channels,hidden_dim,last_dim):
        super(ConvRnnEncoder, self).__init__()
        self.conv_w0 = nn.Conv2d(in_channels,hidden_dim,kernel_size=3,stride=3)
        self.conv_w1 = nn.Conv2d(hidden_dim,last_dim,kernel_size=8,stride=1)
        self.conv_r0 = nn.Conv2d(hidden_dim,hidden_dim,kernel_size=1,stride=1)
        self.linear_r1 = nn.Linear(last_dim,last_dim)

    
    def forward(self, x):
        for i in range(x.shape[1]):
            layer = x[:,i]
            if i == 0:
                h1 = F.relu(self.conv_w0(layer))
                h2 = F.relu(self.conv_w1(h1))
                h2 = torch.squeeze(h2,dim=2)
                h2 = torch.squeeze(h2,dim=2)
            else : 
                h1 = F.relu(self.conv_w0(layer)+self.conv_r0(h1))
                #compute part from hidden layer 
                h2_hidden = F.relu(self.conv_w1(h1))
                h2_hidden = torch.squeeze(h2_hidden,dim=2)
                h2_hidden = torch.squeeze(h2_hidden,dim=2)
                #compute part from last layer
                h2_last = self.linear_r1(h2)
                h2 = F.relu(h2_hidden+h2_last)
        return h1,h2

class ConvRnnDecoder(nn.Module):
    def __init__(self,out_channels,hidden_dim,last_dim,number_layers):
        super(ConvRnnDecoder, self).__init__()
        self.number_layers = number_layers
        self.conv_w0 = nn.ConvTranspose2d(hidden_dim,out_channels,kernel_size=3,stride=3)
        self.conv_w1 = nn.ConvTranspose2d(last_dim,hidden_dim,kernel_size=8,stride=1)
        self.conv_r0 = nn.Conv2d(hidden_dim,hidden_dim,kernel_size=1,stride=1)
        self.linear_r1 = nn.Linear(last_dim,last_dim)

    def forward(self,h1,h2):
        layers = []
        for i in range(self.number_layers+1):
            if i == 0:
                h2 = F.relu(self.linear_r1(h2))
                h1_hidden = self.conv_r0(h1)
                h1_last = self.conv_w1(torch.unsqueeze(torch.unsqueeze(h2,2),2))
                h1 = F.relu(h1_hidden+h1_last)
            else : 
                h2 = F.relu(self.linear_r1(h2))
                h1_hidden = self.conv_r0(h1)
                h1_last = self.conv_w1(torch.unsqueeze(torch.unsqueeze(h2,2),2))
                h1 = F.relu(h1_hidden+h1_last)
                layer = F.elu(self.conv_w0(h1))
                layers.append(layer)
        layers = torch.stack(layers)
        layers = layers.permute(1,0,2,3,4)
        return layers


