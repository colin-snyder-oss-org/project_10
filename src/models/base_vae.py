# src/models/base_vae.py
import torch.nn as nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, input):
        pass

    @abstractmethod
    def decode(self, input):
        pass

    @abstractmethod
    def reparameterize(self, mu, log_var):
        pass

    @abstractmethod
    def forward(self, input):
        pass

    def sample(self, num_samples, current_device):
        pass

    def generate(self, x):
        pass
