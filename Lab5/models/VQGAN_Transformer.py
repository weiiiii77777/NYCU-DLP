import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        zq, indices, _ = self.vqgan.encode(x)
        indices = indices.view(zq.shape[0], -1)
        # zq (batch_size, latent_dim, h*w) 
        # indices (batch_size * latent_dim)
        return zq, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        
        if mode == "cosine" :
            return lambda r : np.cos(r * np.pi / 2)
        elif mode == "linear" :
            return lambda r : 1 - r
        elif mode == "square" :
            return lambda r : 1 - r*r

##TODO2 step1-3:
# https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py
    def forward(self, x):

        # z_indices = ground truth
        # logits = transformer predict the probability of tokens
        zq, z_indices = self.encode_to_z(x)
        # z_indices (batch_size, h*w)
        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        # sample (batch_size, r)
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        # mask (batch_size, h*w)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        # masked_indices (batch_size, h*w)
        a_indices = mask * z_indices + (~mask) * masked_indices
        # a_indices (batch_size, h*w)
        logits = self.transformer(a_indices)
        # logits (batch_size, h*w, 1025)
        return logits, z_indices
    
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, mask_num, ratio):
        
        z_indices[mask_bc] = 1024
        logits = self.transformer(z_indices)
        # logits (batch_size, h*w, 1025)

        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = nn.functional.softmax(logits, -1)
        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)
        # z_indices_predict_prob (batch_size, h*w)
        # z_indices_predict (batch_size, h*w)
        z_indices_predict_prob[~mask_bc] = float('inf')
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob))) # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        _, sorted_indices = torch.sort(confidence)
        z_indices_predict[~mask_bc] = z_indices[~mask_bc]
        mask_bc[:, sorted_indices[:, math.floor(ratio*mask_num):]] = False
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}