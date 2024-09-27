# src/evaluation/evaluator.py
import torch
from src.data.data_loader import get_data_loader
from src.models.hierarchical_vae import HierarchicalVAE
from src.utils.logger import setup_logger
from src.utils.anomaly_detector import detect_anomalies
from src.utils.visualization import plot_latent_space
import os

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = HierarchicalVAE(
            input_dim=config['model']['input_dim'],
            latent_dims=config['model']['latent_dims'],
            hidden_dims=config['model']['hidden_dims']['encoder']
        ).to(self.device)

        # Load the model checkpoint
        checkpoint_path = os.path.join(self.config['training']['checkpoint_path'], f'checkpoint_epoch_{self.config["training"]["epochs"]}.pth')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        # Data loaders
        self.test_loader = get_data_loader(config, mode='test')

        # Logger
        self.logger = setup_logger('Evaluator', 'INFO')

    def evaluate(self):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data = data.to(self.device)
                recon, mu_list, log_var_list = self.model(data)
                loss = self.model.loss_function(recon, data, mu_list, log_var_list)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        self.logger.info(f'====> Test set loss: {avg_loss:.4f}')

    def run_inference(self):
        # Collect latent representations
        all_mu = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data = data.to(self.device)
                mu_list, _, _ = self.model.encode(data)
                mu = mu_list[-1]  # Use the last latent layer
                all_mu.append(mu.cpu())
                # If labels are available, collect them for analysis
                # labels = data['label']
                # all_labels.append(labels)

        # Concatenate all latent representations
        latent_space = torch.cat(all_mu, dim=0)
        # labels = torch.cat(all_labels, dim=0)

        # Detect anomalies
        anomalies = detect_anomalies(latent_space, threshold=self.config['anomaly_detection']['threshold'])

        # Visualize latent space
        plot_latent_space(latent_space, anomalies)
