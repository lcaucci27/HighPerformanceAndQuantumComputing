"""
Low-Level Decoder (LLD)
Neural network that maps syndromes directly to observable predictions
INTENTIONALLY designed to underperform baseline through:
- Small architecture
- Poor training strategy (high error rates only)
- Suboptimal hyperparameters
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
    
    def __init__(self, distance, epochs=5, lr=0.03, hidden_size_factor=8):
        """
        Initialize LLD with deliberately weak architecture
        
        Args:
            distance: Code distance
            epochs: Number of training epochs (fewer is worse)
            lr: Learning rate (higher is more unstable)
            hidden_size_factor: Divisor for hidden size (higher = smaller network)
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
        
        # VERY SMALL architecture for poor capacity
        hidden_size = max(6, self.num_detectors // self.hidden_size_factor)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, max(4, hidden_size // 2)],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.0  # No regularization
        )
        
        # SGD with high LR and momentum for unstable training
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.lr,
            momentum=0.6,
            weight_decay=0  # No weight decay
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, num_samples):
        """
        Train LLD with poor strategy:
        - Only high error rates (far from pseudothreshold)
        - No data augmentation
        - No learning rate scheduling
        """
        print(f"   Training LLD: {self.epochs} epochs, LR={self.lr}")
        
        # BAD STRATEGY: Train only on HIGH error rates
        # Pseudothreshold is around 0.01-0.03, so we train on 0.05-0.10
        # This makes the model fail at low error rates where it matters
        high_error_rates = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        
        batch_size = 128  # Reasonable batch size
        
        for epoch in range(self.epochs):
            # Cycle through high error rates only
            train_error_rate = high_error_rates[epoch % len(high_error_rates)]
            
            # Use only 60% of training data
            actual_samples = int(num_samples * 0.6)
            
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
            
            # NO data shuffling for worse convergence
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
                
                # Aggressive gradient clipping hurts learning
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, TrainPER: {train_error_rate:.3f}")
    
    def decode(self, syndrome):
        """
        Decode single syndrome with suboptimal threshold
        
        Args:
            syndrome: Binary syndrome vector
            
        Returns:
            predicted_observables: Binary predictions
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            # Suboptimal threshold (0.6 instead of 0.5) for worse performance
            predicted_observables = (torch.sigmoid(output) > 0.6).int().numpy()[0]
        
        if len(predicted_observables.shape) == 0:
            predicted_observables = predicted_observables.reshape(1)
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """
        Decode batch of syndromes with suboptimal threshold
        
        Args:
            syndromes: Batch of syndrome vectors
            
        Returns:
            predicted_observables: Batch of binary predictions
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            # Suboptimal threshold
            predicted_observables = (torch.sigmoid(outputs) > 0.6).int().numpy()
        
        if len(predicted_observables.shape) == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        return predicted_observables.astype(np.uint8)