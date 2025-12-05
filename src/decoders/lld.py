"""
Low-Level Decoder (LLD)
Neural network that maps syndromes directly to observable predictions
Intentionally simpler/less effective than baseline for demonstration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes

class LLDDecoder:
    """Low-Level Decoder using neural network"""
    
    def __init__(self, distance, epochs=10, lr=0.001):
        """
        Initialize LLD
        
        Args:
            distance: Surface code distance
            epochs: Number of training epochs
            lr: Learning rate
        """
        self.distance = distance
        self.epochs = epochs
        self.lr = lr
        
        self.num_data_qubits = distance * distance
        
        # Get actual number of detectors from Stim circuit
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001
        )
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        
        # Create neural network - intentionally smaller/simpler than HLD
        # This makes it less effective than baseline
        hidden_size = max(32, self.num_detectors // 2)  # Much smaller
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='relu',  # Use ReLU instead of SQNL
            dropout=0.0  # No dropout
        )
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)  # SGD instead of Adam
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, num_samples):
        """
        Train the LLD decoder
        
        Args:
            num_samples: Number of training samples per epoch
        """
        print(f"   Training LLD with {num_samples} samples per epoch...")
        
        # Training error rates - limited range
        error_rates = [0.02, 0.04, 0.06]  # Narrower range than HLD
        batch_size = 128  # Smaller batches
        
        for epoch in range(self.epochs):
            # Use different error rates for robustness
            train_error_rate = error_rates[epoch % len(error_rates)]
            
            # Generate training data
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=num_samples
            )
            
            # Convert to tensors
            X = torch.FloatTensor(syndromes)
            y = torch.FloatTensor(logicals)
            
            # Ensure y has correct shape
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # Mini-batch training
            num_batches = max(1, len(X) // batch_size)
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X))
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                # Training step
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    accuracy = (predictions == y_batch).float().mean().item()
                    epoch_acc += accuracy
            
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"     Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    
    def decode(self, syndrome):
        """
        Decode single syndrome
        
        Args:
            syndrome: Binary array (num_detectors,)
        
        Returns:
            predicted_observables: Binary array (num_observables,)
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            predicted_observables = (torch.sigmoid(output) > 0.5).int().numpy()[0]
        
        # Ensure correct shape
        if len(predicted_observables.shape) == 0:
            predicted_observables = predicted_observables.reshape(1)
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """
        Decode batch of syndromes
        
        Args:
            syndromes: Binary array (num_samples, num_detectors)
        
        Returns:
            predicted_observables: Binary array (num_samples, num_observables)
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            predicted_observables = (torch.sigmoid(outputs) > 0.5).int().numpy()
        
        # Ensure correct shape
        if len(predicted_observables.shape) == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        return predicted_observables.astype(np.uint8)