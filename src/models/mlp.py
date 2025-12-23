"""
Fully connected Multi-Layer Perceptron
Flexible architecture supporting 2+ hidden layers with SQNL activation
"""

import torch
import torch.nn as nn
from src.models.activations import SQNLEfficient

class MLP(nn.Module):
    """Multi-Layer Perceptron with flexible hidden layers and SQNL support"""
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.0):
        """
        Initialize MLP
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes [h1, h2, ...] (at least 2)
            output_size: Number of output features
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'sqnl')
            dropout: Dropout probability (0.0 = no dropout)
        """
        super(MLP, self).__init__()
        
        assert len(hidden_sizes) >= 2, "Must have at least 2 hidden layers"
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'sqnl':
                layers.append(SQNLEfficient())  # FIXED: Add SQNL support
            else:
                layers.append(nn.ReLU())
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, output_size))
        
        # Combine into sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, input_size)
        
        Returns:
            Output tensor (batch_size, output_size)
        """
        return self.network(x)