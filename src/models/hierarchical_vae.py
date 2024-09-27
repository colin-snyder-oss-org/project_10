# src/models/hierarchical_vae.py
import torch
from torch import nn
from src.models.base_vae import BaseVAE
from torch.nn import functional as F

class HierarchicalVAE(BaseVAE):
    def __init__(self, input_dim, latent_dims, hidden_dims=None):
        super(HierarchicalVAE, self).__init__()

        self.latent_dims = latent_dims
        self.num_layers = len(latent_dims)

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Build Encoder
        modules = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(last_dim, h_dim),
                nn.ReLU()))
            last_dim = h_dim

        self.encoder = nn.Sequential(*modules)

        # Hierarchical Latent Variables
        self.fc_mu = nn.ModuleList()
        self.fc_var = nn.ModuleList()
        for l_dim in latent_dims:
            self.fc_mu.append(nn.Linear(last_dim, l_dim))
            self.fc_var.append(nn.Linear(last_dim, l_dim))
            last_dim = l_dim

        # Build Decoder
        modules = []
        reversed_hidden_dims = hidden_dims[::-1]
        last_dim = latent_dims[-1]
        for h_dim in reversed_hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(last_dim, h_dim),
                nn.ReLU()))
            last_dim = h_dim

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(last_dim, input_dim),
            nn.Sigmoid())

    def encode(self, input):
        result = self.encoder(input)
        mu_list = []
        log_var_list = []
        for fc_mu, fc_var in zip(self.fc_mu, self.fc_var):
            mu = fc_mu(result)
            log_var = fc_var(result)
            mu_list.append(mu)
            log_var_list.append(log_var)
            # Reparameterize at each layer
            z = self.reparameterize(mu, log_var)
            result = z  # Use z as input to the next layer
        return mu_list, log_var_list, result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        mu_list, log_var_list, z = self.encode(input)
        recon = self.decode(z)
        return recon, mu_list, log_var_list

    def loss_function(self, recon, input, mu_list, log_var_list):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, input, reduction='mean')

        # KL Divergence loss for hierarchical latent variables
        kl_loss = 0
        for mu, log_var in zip(mu_list, log_var_list):
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss += kl

        return recon_loss + kl_loss
