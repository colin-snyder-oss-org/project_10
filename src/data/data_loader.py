# src/data/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class NetworkTrafficDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        # Preprocessing steps can be added here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = sample.values.astype('float32')
        return torch.tensor(features)

def get_data_loader(config, mode='train'):
    if mode == 'train':
        data_file = os.path.join(config['data']['processed_data_path'], 'train.csv')
    else:
        data_file = os.path.join(config['data']['processed_data_path'], 'test.csv')

    dataset = NetworkTrafficDataset(data_file)
    data_loader = DataLoader(dataset,
                             batch_size=config['data']['batch_size'],
                             shuffle=(mode=='train'),
                             num_workers=config['data']['num_workers'])
    return data_loader
