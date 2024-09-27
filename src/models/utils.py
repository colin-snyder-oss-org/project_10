# src/models/utils.py
import torch

def calculate_kl_divergence(mu, log_var):
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_div
