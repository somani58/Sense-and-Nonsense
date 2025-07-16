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
    print("[INFO] Initializing loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("[INFO] Loss function and optimizer initialized.")

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print(f"[INFO] Starting training loop for {num_epochs} epochs...")
    for epoch in tqdm(range(num_epochs), desc="Training"):
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

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'[INFO] Epoch [{epoch+1}/{num_epochs}]')
            print(
                f'[INFO] Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(
                f'[INFO] Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
            print('[INFO] ' + '-' * 50)

    print("[INFO] Training loop completed.")
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
    print("[INFO] Plotting training and test loss/accuracy curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
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
    print("[INFO] Training history plot saved to 'results/plots/training_history.png'.")


if __name__ == "__main__":
    print("[INFO] Loading and preprocessing data from '../data/raw/combined_eeg_data.csv'...")
    X, y, label_encoder, scaler = load_and_preprocess_data(
        '../data/raw/combined_eeg_data.csv')
    print(
        f"[INFO] Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
    print("[INFO] Creating data loaders...")
    train_loader, test_loader = create_data_loaders(X, y, batch_size=64)
    print("[INFO] Data loaders created.")
    print("[INFO] Initializing EEGClassifier model...")
    model = EEGClassifier(input_size=64, num_classes=4)
    print("[INFO] Model initialized.")
    print("[INFO] Starting training...")
    history = train_model(model, train_loader, test_loader, num_epochs=100)
    print("[INFO] Training completed.")
    print("[INFO] Plotting training history...")
    plot_training_history(history)
    print("[INFO] Training history plotted and saved.")
    print("[INFO] Saving trained model to 'models/saved_models/eeg_classifier.pth'...")
    torch.save(model.state_dict(), 'models/saved_models/eeg_classifier.pth')
    print("[INFO] Model saved successfully!")
