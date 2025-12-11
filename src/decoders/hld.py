"""
High-Level Decoder (HLD)
Advanced neural network for logical error classification
REDESIGNED with proper training strategy that actually works
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes

class HLDDecoder:
    """High-Level Decoder - optimized to beat baseline"""
    
    def __init__(self, distance, epochs=40, lr=0.001, hidden_size_factor=3):
        """
        Initialize HLD with optimal architecture
        
        Args:
            distance: Code distance
            epochs: Number of training epochs
            lr: Learning rate
            hidden_size_factor: Multiplier for hidden size
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
        
        # Large architecture for superior capacity
        hidden_size = max(64, self.num_detectors * self.hidden_size_factor)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.15
        )
        
        # Adam optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=1e-4
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-5
        )
    
    def train(self, num_samples):
        """
        Train HLD with WORKING strategy:
        - Use MODERATE error rates where class balance is reasonable
        - Mix of rates from 0.01 to 0.05 (not too low!)
        - Progressive curriculum from medium to lower rates
        """
        print(f"   Training HLD: {self.epochs} epochs, LR={self.lr}")
        
        # WORKING STRATEGY: Use moderate error rates with good class balance
        # At these rates, observables flip ~10-40% of the time (learnable!)
        strategic_error_rates = [
            0.015, 0.018, 0.020, 0.022, 0.025,  # Medium rates (good balance)
            0.028, 0.030, 0.035, 0.040, 0.045   # Medium-high rates
        ]
        
        batch_size = 256
        
        for epoch in range(self.epochs):
            # Curriculum: Start with easier medium rates, expand to full range
            if epoch < self.epochs // 3:
                # First third: medium rates only (easier)
                idx_range = range(0, 5)
            elif epoch < 2 * self.epochs // 3:
                # Middle third: medium to medium-high
                idx_range = range(0, 8)
            else:
                # Final third: full range
                idx_range = range(len(strategic_error_rates))
            
            # Cycle through selected rates
            train_error_rate = strategic_error_rates[
                list(idx_range)[epoch % len(idx_range)]
            ]
            
            # Generate training data
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=num_samples
            )
            
            # Convert to tensors
            X = torch.FloatTensor(syndromes)
            y = torch.FloatTensor(logicals)
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # Check class balance for debugging
            positive_rate = y.mean().item()
            
            # Shuffle data
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            
            # Mini-batch training
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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
                      f"LR: {current_lr:.6f}, TrainPER: {train_error_rate:.3f}, "
                      f"PosRate: {positive_rate:.2%}")
    
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