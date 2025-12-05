"""
High-Level Decoder (HLD)
Advanced neural network for logical error classification
Should outperform baseline decoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes

class HLDDecoder:
    """High-Level Decoder: Advanced NN for logical error classification"""
    
    def __init__(self, distance, epochs=10, lr=0.001):
        """
        Initialize HLD
        
        Args:
            distance: Surface code distance
            epochs: Number of training epochs
            lr: Learning rate
        """
        self.distance = distance
        self.epochs = epochs
        self.lr = lr
        
        self.num_data_qubits = distance * distance
        
        # Get actual number of detectors from Stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001
        )
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        
        # Neural network for observable prediction
        # Larger and more sophisticated than LLD
        hidden_size = max(512, self.num_detectors * 8)  # Much larger
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='sqnl',  # Use SQNL activation
            dropout=0.2  # Add dropout for regularization
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # Use focal loss for better handling of imbalanced data
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
    
    def train(self, num_samples):
        """Train the HLD decoder"""
        print(f"   Training HLD with {num_samples} samples per epoch...")
        
        # Training error rates - comprehensive sweep for robustness
        error_rates = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07]
        batch_size = 1024  # Larger batches
        
        for epoch in range(self.epochs):
            # Use different error rates
            train_error_rate = error_rates[epoch % len(error_rates)]
            
            # Generate training data
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=num_samples
            )
            
            # Convert to tensors with label smoothing for better generalization
            X = torch.FloatTensor(syndromes)
            y = torch.FloatTensor(logicals)
            
            # Label smoothing: targets become 0.05 and 0.95 instead of 0 and 1
            y = y * 0.9 + 0.05
            
            # Ensure y has correct shape
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # Mini-batch training
            num_batches = max(1, len(X) // batch_size)
            epoch_loss = 0
            epoch_acc = 0
            
            # Shuffle data
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            
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
            
            # Update learning rate
            self.scheduler.step(avg_loss)
            
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
        # Predict observables with NN
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            predicted_observables = (torch.sigmoid(output / 0.8) > 0.5).int().numpy()[0]
        
        # Ensure correct shape
        if len(predicted_observables.shape) == 0:
            predicted_observables = predicted_observables.reshape(1)
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """Decode batch of syndromes"""
        # Predict observables with NN
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            # Use temperature scaling for better calibration
            predicted_observables = (torch.sigmoid(outputs / 0.8) > 0.5).int().numpy()
        
        # Ensure correct shape
        if len(predicted_observables.shape) == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        return predicted_observables.astype(np.uint8)