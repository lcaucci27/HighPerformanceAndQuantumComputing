"""
Fully connected Multi-Layer Perceptron
2 hidden layers with configurable activation function
"""

import torch
import torch.nn as nn
from src.models.activations import SQNL

class MLP(nn.Module):
    """Multi-Layer Perceptron with 2 hidden layers"""
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='sqnl'):
        """
        Initialize MLP
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes [h1, h2]
            output_size: Number of output features
            activation: Activation function ('sqnl', 'relu', 'tanh')
        """
        super(MLP, self).__init__()
        
        assert len(hidden_sizes) == 2, "Must have exactly 2 hidden layers"
        
        h1, h2 = hidden_sizes
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_size, h1)
        
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(h1, h2)
        
        # Second hidden layer to output
        self.fc3 = nn.Linear(h2, output_size)
        
        # Activation function
        if activation == 'sqnl':
            self.activation = SQNL()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, input_size)
        
        Returns:
            Output tensor (batch_size, output_size)
        """
        # First hidden layer
        x = self.fc1(x)
        x = self.activation(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.activation(x)
        
        # Output layer (no activation, use with appropriate loss)
        x = self.fc3(x)
        
        return x