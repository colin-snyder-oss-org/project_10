from setuptools import setup, find_packages

setup(
    name='hierarchical_vae_anomaly_detection',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'pyyaml',
        'tensorboard',
        'seaborn',
    ],
    author='Your Name',
    author_email='youremail@example.com',
    description='Hierarchical VAE for unsupervised anomaly detection in network security',
    url='https://github.com/yourusername/hierarchical-vae-anomaly-detection',
)
