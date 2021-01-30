import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0")



class DAGMM(nn.Module):
    def __init__(self, hyp):
        super(DAGMM, self).__init__()
        
        layers = []
        layers += [nn.Linear(hyp['input_dim'],hyp['hidden1_dim'])]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hyp['hidden1_dim'],hyp['hidden2_dim'])]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hyp['hidden2_dim'],hyp['hidden3_dim'])]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hyp['hidden3_dim'],hyp['zc_dim'])]

        self.encoder = nn.Sequential(*layers)


        layers = []
        layers += [nn.Linear(hyp['zc_dim'],hyp['hidden3_dim'])]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hyp['hidden3_dim'],hyp['hidden2_dim'])]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hyp['hidden2_dim'],hyp['hidden1_dim'])]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(hyp['hidden1_dim'],hyp['input_dim'])]

        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(hyp['zc_dim']+2,hyp['hidden3_dim'])]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=hyp['dropout'])]        
        layers += [nn.Linear(hyp['hidden3_dim'],hyp['n_gmm'])]
        layers += [nn.Softmax(dim=1)]


        self.estimation = nn.Sequential(*layers)

        
        self.lambda1 = hyp['lambda1']
        self.lambda2 = hyp['lambda2']


    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        
        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = F.pairwise_distance(x, dec,p=2)
        
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)
        
        return enc,dec,z,gamma
    
    @staticmethod
    def reconstruct_error(x, x_hat):   #重构误差
        e = torch.tensor(0.0)
        for i in range(x.shape[0]):
            e += torch.dist(x[i], x_hat[i])
        return e / x.shape[0]
    
    @staticmethod
    def get_gmm_param(gamma, z):
        N = gamma.shape[0]
        ceta = torch.sum(gamma, dim=0) / N  #shape: [n_gmm]
        
        mean = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0)
        mean = mean / torch.sum(gamma, dim=0).unsqueeze(-1)  #shape: [n_gmm, z_dim]
            

        z_mean = (z.unsqueeze(1)- mean.unsqueeze(0))
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mean.unsqueeze(-1) * z_mean.unsqueeze(-2), dim = 0) / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

            
        return ceta, mean, cov
    
    @staticmethod
    def sample_energy(ceta, mean, cov, zi,n_gmm,bs):
        e = torch.tensor(0.0)
        cov_eps = torch.eye(mean.shape[1]) * (1e-12)
#         cov_eps = cov_eps.to(device)
        for k in range(n_gmm):
            miu_k = mean[k].unsqueeze(1)
            d_k = zi - miu_k

            inv_cov = torch.inverse(cov[k] + cov_eps)
            e_k = torch.exp(-0.5 * torch.chain_matmul(torch.t(d_k), inv_cov, d_k))
            e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * cov[k])))
            e_k = e_k * ceta[k]
            e += e_k.squeeze()
        return -torch.log(e)
    
    

    def loss_func(self, x, dec, gamma, z):
        bs,n_gmm = gamma.shape[0],gamma.shape[1]
        
        #1
        recon_error = self.reconstruct_error(x, dec)
        
        #2
        ceta, mean, cov = self.get_gmm_param(gamma, z)
#         ceta = ceta.to(device)
#         mean = mean.to(device)
#         cov = cov.to(device)

        
        #3
        e = torch.tensor(0.0)
        for i in range(z.shape[0]):
            zi = z[i].unsqueeze(1)
            ei = self.sample_energy(ceta, mean, cov, zi,n_gmm,bs)
            e += ei
        
        p = torch.tensor(0.0)
        for k in range(n_gmm):
            cov_k = cov[k]
            p_k = torch.sum(1 / torch.diagonal(cov_k, 0))
            p += p_k
   

        loss = recon_error + (self.lambda1 / z.shape[0]) * e   + self.lambda2 * p
        
        return loss, recon_error, e/z.shape[0], p
