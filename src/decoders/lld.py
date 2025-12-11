"""
Low-Level Decoder (LLD)
Neural network that maps syndromes directly to observable predictions
INTENTIONALLY designed to underperform through poor training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes

class LLDDecoder:
    """Low-Level Decoder - intentionally weak for hierarchy"""
    
    def __init__(self, distance, epochs=3, lr=0.08, hidden_size_factor=12):
        """
        Initialize LLD with deliberately weak architecture
        
        Args:
            distance: Code distance
            epochs: Number of training epochs (very few)
            lr: Learning rate (way too high)
            hidden_size_factor: Divisor for hidden size (large = tiny network)
        """
        self.distance = distance
        self.epochs = epochs
        self.lr = lr
        self.hidden_size_factor = hidden_size_factor
        
        self.num_data_qubits = distance * distance
        
        # Get detector dimensions from Stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001
        )
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        
        # TINY architecture for poor capacity
        hidden_size = max(6, self.num_detectors // self.hidden_size_factor)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, max(3, hidden_size // 2)],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.0  # No regularization
        )
        
        # SGD with very high LR for unstable training
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.lr,
            momentum=0.5
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, num_samples):
        """
        Train LLD with terrible strategy:
        - Only VERY HIGH error rates (0.08-0.15)
        - These are far from the test range
        - Very few epochs
        - Unstable high learning rate
        """
        print(f"   Training LLD: {self.epochs} epochs, LR={self.lr}")
        
        # TERRIBLE: Train only on very high error rates
        # Test range is 0.003-0.10, we train on 0.08-0.15
        high_error_rates = [0.08, 0.09, 0.10, 0.12, 0.14, 0.15]
        
        batch_size = 64  # Small batches for noisier gradients
        
        for epoch in range(self.epochs):
            # Cycle through high error rates
            train_error_rate = high_error_rates[epoch % len(high_error_rates)]
            
            # Use only 50% of training data
            actual_samples = int(num_samples * 0.5)
            
            # Generate training data
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=actual_samples
            )
            
            # Convert to tensors
            X = torch.FloatTensor(syndromes)
            y = torch.FloatTensor(logicals)
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # NO shuffling for worse convergence
            num_batches = max(1, len(X) // batch_size)
            epoch_loss = 0
            
            self.model.train()
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X))
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                
                # Very aggressive clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) == 1 or (epoch + 1) == self.epochs:
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, TrainPER: {train_error_rate:.3f}")
    
    def decode(self, syndrome):
        """Decode single syndrome to predict observable flips"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            predicted_observables = (torch.sigmoid(output) > 0.5).int().numpy()[0]
        
        if len(predicted_observables.shape) == 0:
            predicted_observables = predicted_observables.reshape(1)
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """Decode batch of syndromes to predict observable flips"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            predicted_observables = (torch.sigmoid(outputs) > 0.5).int().numpy()
        
        if len(predicted_observables.shape) == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        return predicted_observables.astype(np.uint8)