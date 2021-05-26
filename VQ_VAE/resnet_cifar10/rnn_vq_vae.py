import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl, kl_divergence

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


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

class VQConvRnnAE(nn.Module):
    def __init__(self,in_channels,hidden_dim,last_dim,num_embeddings, 
            embedding_dim, commitment_cost, decay):
        super().__init__()
        assert ( hidden_dim == embedding_dim)
        self.encoder = ConvRnnEncoder(in_channels,hidden_dim,last_dim*2)
        self.vq_vae = VectorQuantizerEMA(num_embeddings,embedding_dim,commitment_cost,decay)
        self.decoder = ConvRnnDecoder(in_channels,hidden_dim,last_dim, number_layers=19)

    def forward(self,x):
        h1, h2 = self.encoder(x)

        loss_vq, quantized, perplexity, _ = self.vq_vae(h1)

        mu, logvar = h2.chunk(2, dim=1)
        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        h2_sample = q_z_x.rsample()

        x_tilde = self.encoder(quantized,h2_sample)

        return loss_vq, kl_div, x_tilde, perplexity



