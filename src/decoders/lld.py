"""
Low-Level Decoder (LLD)
Neural network that maps syndromes directly to observable predictions
FIXED: Intentionally weak but actually trains
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes

class LLDDecoder:
    """Low-Level Decoder - intentionally weak but realistic"""
    
    def __init__(self, distance, epochs=8, lr=0.01, hidden_size_factor=6):
        """
        Initialize LLD with deliberately weak architecture
        
        Args:
            distance: Code distance
            epochs: Number of training epochs (few)
            lr: Learning rate (moderate)
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
        
        # SMALL architecture for limited capacity
        hidden_size = max(16, self.num_detectors // self.hidden_size_factor)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, max(8, hidden_size // 2)],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.0  # No regularization for faster overfitting
        )
        
        # Adam with moderate LR (will converge but not optimally)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, num_samples):
        """
        Train LLD with suboptimal strategy:
        - Train ONLY on medium-high error rates
        - Limited epochs
        - Smaller network capacity
        - No curriculum or sophisticated techniques
        """
        print(f"   Training LLD: {self.epochs} epochs, LR={self.lr}")
        
        # Train on MEDIUM-HIGH error rates only (poor generalization to low rates)
        # Test range is 0.003-0.10, we train on 0.025-0.055
        medium_high_rates = [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055]
        
        batch_size = 128
        
        for epoch in range(self.epochs):
            # Simple cycling through rates
            train_error_rate = medium_high_rates[epoch % len(medium_high_rates)]
            
            # Use 70% of training data
            actual_samples = int(num_samples * 0.7)
            
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
            
            # Shuffle
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            
            # Mini-batch training
            num_batches = max(1, len(X) // batch_size)
            epoch_loss = 0
            correct = 0
            total = 0
            
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
                
                # Moderate gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                
                # Track accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.numel()
            
            avg_loss = epoch_loss / num_batches
            accuracy = correct / total
            
            if (epoch + 1) == 1 or (epoch + 1) == self.epochs:
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
                      f"Acc: {accuracy:.3f}, TrainPER: {train_error_rate:.3f}")
    
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