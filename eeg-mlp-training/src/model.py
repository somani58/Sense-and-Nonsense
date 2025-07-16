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
