import os
import pandas as pd
import torch
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

    # for directory in directories:
    #     os.makedirs(directory, exist_ok=True)

    print("Project structure created successfully!")


def main():
    print("[INFO] Creating project structure...")
    create_project_structure()
    print("[INFO] Project structure created.")

    print("[INFO] Loading and preprocessing data from '../data/raw/combined_eeg_data.csv'...")
    X, y, label_encoder = load_and_preprocess_data(
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
    print("[INFO] Model saved.")

    print("[INFO] Evaluating model on test data...")
    accuracy, report, cm = evaluate_model(model, test_loader, label_encoder)
    print(f"[INFO] Evaluation complete. Test Accuracy: {accuracy:.4f}")
    print("[INFO] Classification Report:\n" + report)
    print(f"[INFO] Confusion Matrix:\n{cm}")

    print("[INFO] Saving classification report to 'results/reports/classification_report.txt'...")
    with open('results/reports/classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print("[INFO] Classification report saved.")

    print("[INFO] Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
