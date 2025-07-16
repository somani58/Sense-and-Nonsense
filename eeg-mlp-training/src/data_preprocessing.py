import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_preprocess_data(file_path):
    """
    Load and preprocess EEG data for neural network training
    """
    print(f"[INFO] Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    print("[INFO] Parsing complex coefficients...")
    df['real_part'] = df['coeff'].apply(lambda x: complex(x).real)
    df['imag_part'] = df['coeff'].apply(lambda x: complex(x).imag)
    print("[INFO] Complex coefficients parsed.")

    print("[INFO] Creating feature matrix...")
    feature_data = []
    labels = []
    group_count = 0
    for (participant, trial, frequency), group in df.groupby(['participant', 'trial', 'frequency']):
        feature_vector = []
        electrodes = sorted(group['electrode'].unique())
        for electrode in electrodes:
            electrode_data = group[group['electrode'] == electrode]
            if len(electrode_data) > 0:
                feature_vector.extend([
                    electrode_data['real_part'].iloc[0],
                    electrode_data['imag_part'].iloc[0]
                ])
            else:
                feature_vector.extend([0.0, 0.0])
        feature_data.append(feature_vector)
        labels.append(group['condition'].iloc[0])
        group_count += 1
        if group_count % 1000 == 0:
            print(f"[INFO] Processed {group_count} groups...")
    print(f"[INFO] Feature matrix created for {group_count} groups.")

    print("[INFO] Converting features and labels to numpy arrays...")
    X = np.array(feature_data)
    y = np.array(labels)
    print(
        f"[INFO] Feature array shape: {X.shape}, Label array shape: {y.shape}")

    print("[INFO] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"[INFO] Labels encoded. Classes: {list(label_encoder.classes_)}")

    print("[INFO] Normalizing features...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    print("[INFO] Features normalized.")

    return X_normalized, y_encoded, label_encoder


def create_data_loaders(X, y, test_size=0.2, batch_size=64):
    """
    Create train and test data loaders
    """
    print(
        f"[INFO] Splitting data: test_size={test_size}, batch_size={batch_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(
        f"[INFO] Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples.")

    print("[INFO] Creating EEGDataset objects...")
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    print("[INFO] Creating DataLoader objects...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    print("[INFO] DataLoaders created.")

    return train_loader, test_loader
