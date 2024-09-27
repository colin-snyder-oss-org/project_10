# src/data/preprocess.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(config):
    raw_data_path = config['data']['raw_data_path']
    processed_data_path = config['data']['processed_data_path']
    os.makedirs(processed_data_path, exist_ok=True)

    # Load raw data
    data = pd.read_csv(raw_data_path)

    # Data cleaning and preprocessing steps
    # Example: Handle missing values
    data = data.dropna()

    # Feature selection or extraction
    # Example: Selecting numerical features
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    data = data[numerical_features]

    # Standardization
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Split into train and test sets
    train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

    # Save processed data
    pd.DataFrame(train_data).to_csv(os.path.join(processed_data_path, 'train.csv'), index=False)
    pd.DataFrame(test_data).to_csv(os.path.join(processed_data_path, 'test.csv'), index=False)

if __name__ == "__main__":
    from src.config import get_config
    config = get_config('configs/config.yaml')
    preprocess_data(config)
