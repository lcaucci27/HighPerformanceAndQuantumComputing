"""
Fully connected Multi-Layer Perceptron
2 hidden layers with configurable activation function
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """Multi-Layer Perceptron with 2 hidden layers"""
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.0):
        """
        Initialize MLP
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes [h1, h2]
            output_size: Number of output features
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout probability (0.0 = no dropout)
        """
        super(MLP, self).__init__()
        
        assert len(hidden_sizes) == 2, "Must have exactly 2 hidden layers"
        
        h1, h2 = hidden_sizes
        
        # Layers
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output_size)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # Batch normalization for better training
        self.bn1 = nn.BatchNorm1d(h1)
        self.bn2 = nn.BatchNorm1d(h2)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
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
        # First hidden layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.fc3(x)
        
        return x