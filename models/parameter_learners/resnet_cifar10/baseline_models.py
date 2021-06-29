# Here we want to tae an input from any layer and 
# try to reconstruct it using a vae and vq-vae 
# We ignore the layer depth and thus pad to max size

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import normalization
from models.VQ_VAE.basic_gc.vq_vae import VectorQuantizer, VectorQuantizerEMA
from .types_ import *

class ResidualIntermediateBlock(nn.Module):
    '''a building element to have before (encoder) or after (decoder) down/upsamplings 
    to make the network deeper. They keep the spatial dimensions of the input, so input dim = output dim'''
    def __init__(self, in_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out

class LayerVAEresC10(nn.Module):

    def __init__(self,
                in_channels: int,
                latent_dim: int,
                hidden_dims: list = None,
                pre_interm_layers: int = 1,
                interm_layers: int = 1,
                sqrt_number_kernels: int = 8,
                **kwargs) -> None:
        super().__init__()

        self.pre_int_layers = pre_interm_layers > 0
        self.int_layers = interm_layers > 0
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [128,256,512]
        
        #pre-intermediate layers 
        modules = []
        for _ in range(pre_interm_layers):
            modules.append(ResidualIntermediateBlock(in_channels,kernel_size=3,padding=1))
        self.pre_intermediate_layer = nn.Sequential(*modules)
        
        #So called Embedding layer, which tries to detect the convolutions in the layers
        self.embedding_layer = nn.Sequential(
                                nn.Conv2d(in_channels,hidden_dims[0],
                                    kernel_size=3, stride=3),
                                nn.BatchNorm2d(hidden_dims[0]))
        
        #Intermediate layer which manipulates the "embedded" kernels
        modules = []
        for _ in range(interm_layers):
            modules.append(ResidualIntermediateBlock(hidden_dims[0], kernel_size=3,padding=1))
        self.enc_intermediate_layer = nn.Sequential(*modules)
        
        # Combiantion layer, or downsize layer nr 2  
        self.combination_layer = nn.Sequential(
                                    nn.Conv2d(hidden_dims[0],hidden_dims[1],
                                        kernel_size=sqrt_number_kernels, stride=1),
                                    nn.BatchNorm2d(hidden_dims[1]),
                                    nn.Tanh())

        self.fc_enc = nn.Sequential(
                            nn.Linear(hidden_dims[1], hidden_dims[2]),
                            nn.LeakyReLU())          
        self.fc_mu = nn.Linear(hidden_dims[2], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[2], latent_dim) 

        #Decoder 
        self.fc_dec = nn.Sequential(
                            nn.Linear(latent_dim, hidden_dims[1]),
                            nn.LeakyReLU())   

        self.decombination_layer = nn.Sequential(
                                    nn.ConvTranspose2d(hidden_dims[1],hidden_dims[0],
                                        kernel_size=sqrt_number_kernels, stride=1),
                                    nn.BatchNorm2d(hidden_dims[0]),
                                    nn.Tanh())   
        
        modules = []
        for _ in range(interm_layers):
            modules.append(ResidualIntermediateBlock(hidden_dims[0],kernel_size=3,padding=1))
        self.dec_intermediate = nn.Sequential(*modules)

        self.dec_embedding_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[0],in_channels,
                                    kernel_size=3, stride=3),
                                nn.BatchNorm2d(in_channels))

        #pre-intermediate layers 
        modules = []
        for _ in range(pre_interm_layers):
            modules.append(ResidualIntermediateBlock(in_channels,kernel_size=3,padding=1))
        self.post_intermediate_layer = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
                                nn.Conv2d(in_channels, in_channels,
                                    kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(in_channels),
                                nn.Tanh())
        

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = input
        if self.pre_int_layers:
            result = self.pre_intermediate_layer(result) + result
        result = self.embedding_layer(result)
        if self.int_layers:
            result = self.enc_intermediate_layer(result) + result
        result = self.combination_layer(result)
        result = torch.squeeze(result)
        result = self.fc_enc(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.fc_dec(z)
        result = torch.unsqueeze(result, dim=2)
        result = torch.unsqueeze(result, dim=2)
        result = self.decombination_layer(result)
        if self.int_layers:
            result = self.dec_intermediate(result) + result
        result = self.dec_embedding_layer(result)
        if self.pre_int_layers:
            result = self.post_intermediate_layer(result) + result
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                    mask,
                    *args,
                    **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        # mask out the reconstructed values which are not taken into account
        recons = recons * mask
        recons_loss =F.mse_loss(recons, input, reduction='sum')
        recons_loss = recons_loss * (1/torch.sum(mask))

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class LayerCVAEresC10(LayerVAEresC10):

    def __init__(self, in_channels: int, 
                latent_dim: int, 
                hidden_dims: list, 
                pre_interm_layers: int, 
                interm_layers: int, 
                sqrt_number_kernels: int, 
                number_layers: int,
                number_archs: int, 
                input_size: int = 24,
                **kwargs) -> None:

        super().__init__(in_channels, latent_dim, hidden_dims=hidden_dims, 
        pre_interm_layers=pre_interm_layers, interm_layers=interm_layers, 
        sqrt_number_kernels=sqrt_number_kernels, **kwargs)
        
        self.input_size = input_size
        self.embed_layer = nn.Linear(number_layers, input_size**2)
        self.embed_arch = nn.Linear(number_archs, input_size**2)
        #a layer to scale down the number of slices to use structure
        self.first_layer = nn.Conv2d(in_channels + 2,in_channels,
            kernel_size=1,padding=0)
        
        self.fc_dec = nn.Sequential(
                            nn.Linear(latent_dim + number_layers + number_archs, 
                            hidden_dims[1]), nn.LeakyReLU()) 
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.first_layer(input)
        if self.pre_int_layers:
            result = self.pre_intermediate_layer(result) + result
        result = self.embedding_layer(result)
        if self.int_layers:
            result = self.enc_intermediate_layer(result) + result
        result = self.combination_layer(result)
        result = torch.squeeze(result)
        result = self.fc_enc(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def forward(self, input, layer, arch):

        layer = layer.float()
        embedded_layer = self.embed_layer(layer)
        embedded_layer = embedded_layer.view(-1, self.input_size, self.input_size).unsqueeze(1)

        arch = arch.float()
        embedded_arch = self.embed_arch(arch)
        embedded_arch = embedded_arch.view(-1, self.input_size, self.input_size).unsqueeze(1)

        x = torch.cat([input, embedded_arch, embedded_layer], dim = 1)

        mu, log_var = self.encode(input)

        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, arch, layer], dim = 1)

        return  [self.decode(z), input, mu, log_var]

    def sample(self,
               num_samples: int,
               current_device: int, 
               layer: int,
               arch: int) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        layer = layer.float()
        arch = arch.float()

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)
        z = torch.cat([z, layer, arch], dim=1)
        samples = self.decode(z)
        return samples

class LayerVQVAEresC10(nn.Module):

    def __init__(self,
                in_channels: int,
                embedding_dim: int,
                num_embeddings: int,
                commitment_cost: float,
                decay: float,
                hidden_dims: list = None,
                pre_interm_layers: int = 1,
                interm_layers: int = 1,
                sqrt_number_kernels: int = 8,
                **kwargs) -> None:
        super().__init__()

        self.pre_int_layers = pre_interm_layers > 0
        self.int_layers = interm_layers > 0

        if hidden_dims is None:
            hidden_dims = [256]
        
        #pre-intermediate layers 
        modules = []
        for _ in range(pre_interm_layers):
            modules.append(ResidualIntermediateBlock(in_channels, kernel_size=3, padding=1))
        self.pre_intermediate_layer = nn.Sequential(*modules)
        
        #So called Embedding layer, which tries to detect the convolutions in the layers
        self.embedding_layer = nn.Sequential(
                                nn.Conv2d(in_channels, hidden_dims[0],
                                    kernel_size=3, stride=3),
                                nn.BatchNorm2d(hidden_dims[0]))
        
        #Intermediate layer which manipulates the "embedded" kernels
        modules = []
        for _ in range(interm_layers):
            modules.append(ResidualIntermediateBlock(hidden_dims[0], kernel_size=3,padding=1))
        self.enc_intermediate_layer = nn.Sequential(*modules)
        
        self._pre_vq_conv = nn.Conv2d(hidden_dims[0],embedding_dim,kernel_size=3,stride=1,padding=1)

        if decay > 0.0 :
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)  
      
        #Decoder
        self._post_vq_conv = nn.Conv2d(embedding_dim,hidden_dims[0],kernel_size=1,padding=0)
        modules = []
        for _ in range(interm_layers):
            modules.append(ResidualIntermediateBlock(hidden_dims[0],kernel_size=3,padding=1))
        self.dec_intermediate = nn.Sequential(*modules)

        self.dec_embedding_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[0],in_channels,
                                    kernel_size=3, stride=3),
                                nn.BatchNorm2d(in_channels))

        #pre-intermediate layers 
        modules = []
        for _ in range(pre_interm_layers):
            modules.append(ResidualIntermediateBlock(in_channels,kernel_size=3,padding=1))
        self.post_intermediate_layer = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
                                nn.Conv2d(in_channels, in_channels,
                                    kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(in_channels),
                                nn.Tanh())
        

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = input
        if self.pre_int_layers:
            result = self.pre_intermediate_layer(result) + result
        result = self.embedding_layer(result)
        if self.int_layers:
            result = self.enc_intermediate_layer(result) + result
        result = self._pre_vq_conv(result)
        loss, quantized, perplexity, _ = self._vq_vae(result)
        return [loss, quantized, perplexity]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self._post_vq_conv(z)

        if self.int_layers:
            result = self.dec_intermediate(result) + result
        result = self.dec_embedding_layer(result)
        if self.pre_int_layers:
            result = self.post_intermediate_layer(result) + result
        result = self.final_layer(result)
        return result

    def forward(self, input):
        vq_loss, quantized, perplexity = self.encode(input)
        recon = self.decode(quantized)
        return  vq_loss, recon, perplexity

    def loss_function(self,
                    input,
                    mask,
                    recons,
                    vq_loss
                    ) -> dict:

        # mask out the reconstructed values which are not taken into account
        recons = recons * mask
        recons_loss = F.mse_loss(recons, input, reduction='sum')
        recons_loss = recons_loss * (1/torch.sum(mask))

        loss = recons_loss + vq_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'vq_loss':vq_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[1]

class LayerCVQVAEresC10(LayerVQVAEresC10):

    def __init__(self, 
                in_channels: int, 
                embedding_dim: int, 
                num_embeddings: int, 
                commitment_cost: float, 
                decay: float, 
                hidden_dims: list, 
                pre_interm_layers: int, 
                interm_layers: int, 
                sqrt_number_kernels: int, 
                number_archs: int,
                number_layers:int, 
                input_size:int ,
                **kwargs) -> None:

        super().__init__(in_channels, embedding_dim, num_embeddings, 
            commitment_cost, decay, hidden_dims=hidden_dims, pre_interm_layers=pre_interm_layers, 
            interm_layers=interm_layers, sqrt_number_kernels=sqrt_number_kernels, **kwargs)

        if hidden_dims is None:
            hidden_dims = [256]

        self.input_size = input_size
        self.input_size_small = (input_size/3)**2
        #layer to get conditional in encoding 
        self.first_layer = nn.Conv2d(in_channels+2,in_channels,kernel_size=1,padding=0)
        self.embed_layer_big = nn.Linear(number_layers, input_size**2)
        self.embed_arch_big = nn.Linear(number_archs, input_size**2)

        #layers to get conditional in decoding 
        self.embed_layer_small = nn.Linear(number_layers,  self.input_size_small)
        self.embed_arch_small = nn.Linear(number_archs,  self.input_size_small)
        self._post_vq_conv = nn.Conv2d(embedding_dim+2,hidden_dims[0],kernel_size=1,padding=0)
    
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.first_layer(input)
        if self.pre_int_layers:
            result = self.pre_intermediate_layer(result) + result
        result = self.embedding_layer(result)
        if self.int_layers:
            result = self.enc_intermediate_layer(result) + result
        result = self._pre_vq_conv(result)
        loss, quantized, perplexity, _ = self._vq_vae(result)
        return [loss, quantized, perplexity]

    def forward(self, input, arch, layer):
        # embedd labels and append them to input 
        arch = arch.float()
        embedded_arch = self.embed_arch_big(arch)
        embedded_arch = embedded_arch.view(-1,self.input_size,self.input_size)

        layer = layer.float()
        embedded_layer = self.embed_layer_big(layer)
        embedded_layer = embedded_layer.view(-1,self.input_size,self.input_size)

        #get encoding 
        x = torch.cat([input, embedded_arch, embedded_layer],dim=1)
        vq_loss, quantized, perplexity = self.encode(x)

        #embed layers in small format to concatenate to codebook vectors
        embedded_arch = self.embed_arch_small(arch)
        embedded_arch = embedded_arch.view(-1,self.input_size_small,self.input_size_small)
        embedded_layer = self.embed_layer_small(layer)
        embedded_layer = embedded_layer.view(-1,self.input_size_small,self.input_size_small)

        z = torch.cat([quantized, embedded_arch, embedded_layer], dim = 1)
        recon = self.decode(z)
        return  vq_loss, recon, perplexity

    def sample(self,
               num_samples:int,
               current_device: int, 
               arch: int,
               layer: int,
               **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        arch = arch.float()
        layer = layer.float()
        embedded_arch = self.embed_arch_small(arch)
        embedded_arch = embedded_arch.view(-1,self.input_size_small,self.input_size_small)
        embedded_layer = self.embed_layer_small(layer)
        embedded_layer = embedded_layer.view(-1,self.input_size_small,self.input_size_small)

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = torch.cat([z, embedded_arch, embedded_layer])
        
        z = z.to(current_device)

        samples = self.decode(z)
        return samples