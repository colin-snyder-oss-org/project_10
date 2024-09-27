# src/training/trainer.py
import torch
from torch.optim import Adam
from tqdm import tqdm
from src.models.hierarchical_vae import HierarchicalVAE
from src.data.data_loader import get_data_loader
from src.utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = HierarchicalVAE(
            input_dim=config['model']['input_dim'],
            latent_dims=config['model']['latent_dims'],
            hidden_dims=config['model']['hidden_dims']['encoder']
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = Adam(self.model.parameters(), lr=config['model']['learning_rate'])

        # Data loaders
        self.train_loader = get_data_loader(config, mode='train')

        # Logger
        self.logger = setup_logger('Trainer', 'INFO')
        self.writer = SummaryWriter(log_dir=config['logging']['log_dir'])

        # Training parameters
        self.epochs = config['training']['epochs']
        self.kl_annealing = config['model']['kl_annealing']['enabled']
        self.kl_annealing_epochs = config['model']['kl_annealing']['epochs']

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            for batch_idx, data in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon, mu_list, log_var_list = self.model(data)
                loss = self.model.loss_function(recon, data, mu_list, log_var_list)

                # KL Annealing
                if self.kl_annealing:
                    kl_weight = min(1.0, epoch / self.kl_annealing_epochs)
                    loss *= kl_weight

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                if batch_idx % self.config['training']['log_interval'] == 0:
                    self.logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                    self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + batch_idx)

            avg_loss = train_loss / len(self.train_loader)
            self.logger.info(f'====> Epoch {epoch} Average loss: {avg_loss:.4f}')
            self.writer.add_scalar('Loss/avg_train_loss', avg_loss, epoch)

            # Save checkpoint
            checkpoint_path = os.path.join(self.config['training']['checkpoint_path'], f'checkpoint_epoch_{epoch}.pth')
            os.makedirs(self.config['training']['checkpoint_path'], exist_ok=True)
            torch.save(self.model.state_dict(), checkpoint_path)
