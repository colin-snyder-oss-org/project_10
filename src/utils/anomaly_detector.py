# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_latent_space(latent_space, anomalies=None):
    if latent_space.shape[1] >= 2:
        z = latent_space.numpy()
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=z[:,0], y=z[:,1], hue=anomalies)
        plt.title('Latent Space Visualization')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()
    else:
        print("Latent space has less than 2 dimensions, cannot plot.")
