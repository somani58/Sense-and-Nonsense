# EEG MLP Training

This project contains code and resources for training a Multi-Layer Perceptron (MLP) on EEG data.

# EEG Neural Network Training Guide

## Project Overview

This guide will help you create a Multi-Layer Perceptron (MLP) neural network to classify EEG data based on Fourier coefficients. The network will distinguish between 4 conditions using processed EEG signals from 32 electrodes.

## Project Structure

```
eeg_neural_network/
├── data/
│   ├── raw/
│   │   └── eeg_data.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── models/
│   └── saved_models/
├── results/
│   ├── plots/
│   └── reports/
├── requirements.txt
└── README.md
```

## Step 1: Environment Setup

### 1.1 Create Project Directory

```bash
mkdir eeg_neural_network
cd eeg_neural_network
```

### 1.2 Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate


```

### 1.3 Install Required Packages

Create `requirements.txt`:

```txt
# Deep Learning Framework (Latest versions as of July 2025)
torch>=2.6.0                # Latest stable version released Jan 2025
torchvision>=0.21.0         # Compatible with torch 2.6.0

# Core Data Science Libraries
numpy>=1.24.0               # Latest stable, avoid breaking changes
pandas>=2.0.0               # Major version with performance improvements
matplotlib>=3.7.0           # Latest stable plotting library
seaborn>=0.12.0             # Latest statistical visualization

# Machine Learning & Analysis
scikit-learn>=1.3.0         # Latest stable ML library
shap>=0.43.0                # Latest explainability library

# Development & Utilities
jupyter>=1.0.0              # Notebook environment
tqdm>=4.65.0                # Progress bars
```

Install packages:

```bash
pip install -r requirements.txt
```

## Step 2: Data Understanding

### 2.1 Data Structure

Your data contains:

- **participant**: Subject ID (S10, S11, etc.)
- **frequency**: Time point (0.260416666666667 to ~3.5 seconds)
- **electrode**: 32 EEG electrodes (C3, C4, F3, F4, etc.)
- **trial**: Trial number
- **coeff**: Complex Fourier coefficient (real + imaginary parts)
- **condition**: 4 conditions (GN, GS, UGN, UGS)

### 2.2 Network Architecture Design

```
Input Layer: 64 neurons (32 electrodes × 2 components each)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons (ReLU activation)
Output Layer: 4 neurons (softmax activation for 4 conditions)
```

## Step 3: Data Preprocessing

### 3.1 Create data_preprocessing.py

```python
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
    # Load data
    df = pd.read_csv(file_path)

    # Parse complex coefficients
    df['real_part'] = df['coeff'].apply(lambda x: complex(x).real)
    df['imag_part'] = df['coeff'].apply(lambda x: complex(x).imag)

    # Create feature matrix
    # Group by participant, trial, and frequency to get electrode readings
    feature_data = []
    labels = []

    for (participant, trial, frequency), group in df.groupby(['participant', 'trial', 'frequency']):
        # Create feature vector: [real_C3, imag_C3, real_C4, imag_C4, ...]
        feature_vector = []

        # Sort electrodes to ensure consistent ordering
        electrodes = sorted(group['electrode'].unique())

        for electrode in electrodes:
            electrode_data = group[group['electrode'] == electrode]
            if len(electrode_data) > 0:
                feature_vector.extend([
                    electrode_data['real_part'].iloc[0],
                    electrode_data['imag_part'].iloc[0]
                ])
            else:
                feature_vector.extend([0.0, 0.0])  # Fill missing electrodes

        feature_data.append(feature_vector)
        labels.append(group['condition'].iloc[0])

    # Convert to numpy arrays
    X = np.array(feature_data)
    y = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y_encoded, label_encoder, scaler

def create_data_loaders(X, y, test_size=0.2, batch_size=64):
    """
    Create train and test data loaders
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Create datasets
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
```

## Step 4: Model Architecture

### 4.1 Create model.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size1=128, hidden_size2=64, num_classes=4):
        super(EEGClassifier, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
```

## Step 5: Training Script

### 5.1 Create train.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data, create_data_loaders
from model import EEGClassifier

def train_model(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the neural network
    """
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total

        # Store history
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
            print('-' * 50)

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['test_accuracies'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('results/plots/training_history.png')
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, label_encoder, scaler = load_and_preprocess_data('data/raw/eeg_data.csv')

    # Create data loaders
    train_loader, test_loader = create_data_loaders(X, y, batch_size=64)

    # Initialize model
    model = EEGClassifier(input_size=64, num_classes=4)

    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, test_loader, num_epochs=100)

    # Plot results
    plot_training_history(history)

    # Save model
    torch.save(model.state_dict(), 'models/saved_models/eeg_classifier.pth')
    print("Model saved successfully!")
```

## Step 6: Evaluation Script

### 6.1 Create evaluate.py

```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import shap
from model import EEGClassifier
from data_preprocessing import load_and_preprocess_data, create_data_loaders

def evaluate_model(model, test_loader, label_encoder):
    """
    Evaluate the trained model
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/plots/confusion_matrix.png')
    plt.show()

    return accuracy, report, cm

def explain_predictions(model, X_test, electrode_names):
    """
    Use SHAP to explain model predictions
    """
    # Create explainer
    explainer = shap.Explainer(model, X_test[:100])  # Use subset for efficiency
    shap_values = explainer(X_test[:100])

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], feature_names=electrode_names)
    plt.savefig('results/plots/shap_summary.png')
    plt.show()

    return shap_values

def analyze_electrode_importance(shap_values, electrode_names):
    """
    Analyze which electrodes are most important
    """
    # Calculate mean absolute SHAP values for each feature
    mean_shap = np.mean(np.abs(shap_values.values), axis=0)

    # Create DataFrame for analysis
    importance_df = pd.DataFrame({
        'electrode': electrode_names,
        'importance': mean_shap
    })

    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Plot top 20 most important electrodes
    plt.figure(figsize=(12, 8))
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['electrode'])
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title('Top 20 Most Important Electrodes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/plots/electrode_importance.png')
    plt.show()

    return importance_df

if __name__ == "__main__":
    # Load data
    X, y, label_encoder, scaler = load_and_preprocess_data('data/raw/eeg_data.csv')
    train_loader, test_loader = create_data_loaders(X, y)

    # Load trained model
    model = EEGClassifier()
    model.load_state_dict(torch.load('models/saved_models/eeg_classifier.pth'))

    # Evaluate model
    accuracy, report, cm = evaluate_model(model, test_loader, label_encoder)

    # Create electrode names for interpretation
    electrode_names = []
    for i in range(32):  # Assuming 32 electrodes
        electrode_names.extend([f'electrode_{i}_real', f'electrode_{i}_imag'])

    # Explain predictions
    X_test = torch.FloatTensor(X[-100:])  # Use last 100 samples
    shap_values = explain_predictions(model, X_test, electrode_names)

    # Analyze electrode importance
    importance_df = analyze_electrode_importance(shap_values, electrode_names)

    print("Top 10 Most Important Electrodes:")
    print(importance_df.head(10))
```

## Step 7: Running the Complete Pipeline

### 7.1 Create main.py

```python
import os
import pandas as pd
from data_preprocessing import load_and_preprocess_data, create_data_loaders
from model import EEGClassifier
from train import train_model, plot_training_history
from evaluate import evaluate_model, explain_predictions, analyze_electrode_importance

def create_project_structure():
    """Create project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'src',
        'models/saved_models',
        'results/plots',
        'results/reports',
        'notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Project structure created successfully!")

def main():
    # Create project structure
    create_project_structure()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, label_encoder, scaler = load_and_preprocess_data('data/raw/eeg_data.csv')

    # Create data loaders
    train_loader, test_loader = create_data_loaders(X, y, batch_size=64)

    # Initialize model
    model = EEGClassifier(input_size=64, num_classes=4)

    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, test_loader, num_epochs=100)

    # Plot training history
    plot_training_history(history)

    # Save model
    torch.save(model.state_dict(), 'models/saved_models/eeg_classifier.pth')

    # Evaluate model
    accuracy, report, cm = evaluate_model(model, test_loader, label_encoder)

    # Save results
    with open('results/reports/classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
```

## Step 8: Running Instructions

### 8.1 Prepare Your Data

1. Place your EEG data CSV file in `data/raw/eeg_data.csv`
2. Ensure the CSV has columns: participant, frequency, electrode, trial, coeff, condition

### 8.2 Run the Complete Pipeline

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Run the main script
cd src
python main.py
```

### 8.3 Monitor Training

- Watch the training progress in the terminal
- Check the generated plots in `results/plots/`
- Review the classification report in `results/reports/`

## Step 9: Interpretation Guide

### 9.1 Understanding Results

- **Accuracy**: Overall classification performance
- **Confusion Matrix**: Which conditions are confused with each other
- **SHAP Values**: Which electrodes contribute most to predictions

### 9.2 Biological Interpretation

- **High GN-GS confusion**: Semantics might be more important
- **High GN-UGN confusion**: Syntax might be more important
- **Electrode importance**: Shows which brain regions are most active

### 9.3 Next Steps

1. **Hyperparameter tuning**: Adjust learning rate, batch size, network architecture
2. **Cross-validation**: Implement participant-wise cross-validation
3. **Feature engineering**: Try different time windows or frequency bands
4. **Advanced models**: Consider CNN or RNN architectures

## Troubleshooting

### Common Issues:

1. **Memory errors**: Reduce batch size or use data loading in chunks
2. **Overfitting**: Increase dropout, add regularization, or reduce model complexity
3. **Poor accuracy**: Check data preprocessing, try different architectures
4. **Installation issues**: Use conda instead of pip for some packages

### Performance Tips:

- Use GPU if available: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Monitor training with early stopping
- Use learning rate scheduling


