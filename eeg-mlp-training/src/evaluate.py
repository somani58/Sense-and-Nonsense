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
    print("[INFO] Evaluating model...")
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    print(
        f"[INFO] Predictions and labels collected. Total samples: {len(all_labels)}")
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"[INFO] Test Accuracy: {accuracy:.4f}")
    class_names = label_encoder.classes_
    report = classification_report(
        all_labels, all_predictions, target_names=class_names)
    print("[INFO] Classification Report:\n" + report)
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"[INFO] Confusion Matrix:\n{cm}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/plots/confusion_matrix.png')
    print("[INFO] Confusion matrix plot saved to 'results/plots/confusion_matrix.png'.")
    plt.show()
    return accuracy, report, cm


def explain_predictions(model, X_test, electrode_names):
    """
    Use SHAP to explain model predictions
    """
    print("[INFO] Explaining predictions with SHAP...")
    explainer = shap.Explainer(model, X_test[:100])
    shap_values = explainer(X_test[:100])
    print("[INFO] SHAP values computed.")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], feature_names=electrode_names)
    plt.savefig('results/plots/shap_summary.png')
    print("[INFO] SHAP summary plot saved to 'results/plots/shap_summary.png'.")
    plt.show()
    return shap_values


def analyze_electrode_importance(shap_values, electrode_names):
    """
    Analyze which electrodes are most important
    """
    print("[INFO] Analyzing electrode importance using SHAP values...")
    mean_shap = np.mean(np.abs(shap_values.values), axis=0)
    importance_df = pd.DataFrame({
        'electrode': electrode_names,
        'importance': mean_shap
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("[INFO] Top 10 electrodes by importance:")
    print(importance_df.head(10))
    plt.figure(figsize=(12, 8))
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['electrode'])
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title('Top 20 Most Important Electrodes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/plots/electrode_importance.png')
    print("[INFO] Electrode importance plot saved to 'results/plots/electrode_importance.png'.")
    plt.show()
    return importance_df


if __name__ == "__main__":
    print("[INFO] Loading and preprocessing data from 'data/raw/eeg_data.csv'...")
    X, y, label_encoder, scaler = load_and_preprocess_data(
        'data/raw/eeg_data.csv')
    print(
        f"[INFO] Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
    print("[INFO] Creating data loaders...")
    train_loader, test_loader = create_data_loaders(X, y)
    print("[INFO] Data loaders created.")
    print("[INFO] Loading trained EEGClassifier model...")
    model = EEGClassifier()
    model.load_state_dict(torch.load('models/saved_models/eeg_classifier.pth'))
    print("[INFO] Model loaded.")
    print("[INFO] Evaluating model...")
    accuracy, report, cm = evaluate_model(model, test_loader, label_encoder)
    print("[INFO] Creating electrode names for interpretation...")
    electrode_names = []
    for i in range(32):  # Assuming 32 electrodes
        electrode_names.extend([f'electrode_{i}_real', f'electrode_{i}_imag'])
    print(f"[INFO] Electrode names created: {len(electrode_names)} features.")
    print("[INFO] Explaining predictions...")
    X_test = torch.FloatTensor(X[-100:])  # Use last 100 samples
    shap_values = explain_predictions(model, X_test, electrode_names)
    print("[INFO] Analyzing electrode importance...")
    importance_df = analyze_electrode_importance(shap_values, electrode_names)
    print("[INFO] Top 10 Most Important Electrodes:")
    print(importance_df.head(10))
