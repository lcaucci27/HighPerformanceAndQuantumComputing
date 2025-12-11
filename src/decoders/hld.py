"""
High-Level Decoder (HLD)
Advanced neural network for logical error classification
OPTIMIZED to outperform baseline decoder through:
- Large architecture with good capacity
- Strategic training on relevant error rates
- Optimal hyperparameters and regularization
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
    
    def __init__(self, distance, epochs=50, lr=0.0008, hidden_size_factor=4):
        """
        Initialize HLD with optimal architecture
        
        Args:
            distance: Code distance
            epochs: Number of training epochs (more is better)
            lr: Learning rate (lower for stability)
            hidden_size_factor: Multiplier for hidden size (lower = larger network)
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
        
        # LARGE architecture for superior capacity
        hidden_size = max(128, self.num_detectors * self.hidden_size_factor)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.2  # Good regularization
        )
        
        # Adam optimizer with optimal hyperparameters
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=1e-5,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler for fine-tuning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8,
            min_lr=1e-6
        )
    
    def train(self, num_samples):
        """
        Train HLD with strategic curriculum:
        - Focus on LOW to MEDIUM error rates (around pseudothreshold)
        - Progressive training from easy to hard
        - Data shuffling and augmentation
        """
        print(f"   Training HLD: {self.epochs} epochs, LR={self.lr}")
        
        # STRATEGIC: Focus on error rates around pseudothreshold (0.005-0.04)
        # This is where we need to beat the baseline
        strategic_error_rates = [
            0.002, 0.003, 0.004, 0.005, 0.007, 0.010,
            0.012, 0.015, 0.020, 0.025, 0.030, 0.040
        ]
        
        batch_size = 256  # Larger batches for stable gradients
        
        for epoch in range(self.epochs):
            # Curriculum learning: start with easier (medium) rates, then expand
            if epoch < self.epochs // 3:
                # First third: medium error rates
                idx_range = range(5, 9)
            elif epoch < 2 * self.epochs // 3:
                # Middle third: low to medium rates
                idx_range = range(2, 10)
            else:
                # Final third: full range with emphasis on critical region
                idx_range = range(len(strategic_error_rates))
            
            # Select error rate with emphasis on critical region
            train_error_rate = strategic_error_rates[
                list(idx_range)[epoch % len(idx_range)]
            ]
            
            # Use full training data
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=num_samples
            )
            
            # Convert to tensors
            X = torch.FloatTensor(syndromes)
            y = torch.FloatTensor(logicals)
            
            # Light label smoothing for better calibration
            y = y * 0.95 + 0.025
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # IMPORTANT: Shuffle data for better convergence
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
                
                # Moderate gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, TrainPER: {train_error_rate:.3f}")
    
    def decode(self, syndrome):
        """
        Decode single syndrome with optimal threshold
        
        Args:
            syndrome: Binary syndrome vector
            
        Returns:
            predicted_observables: Binary predictions
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            # Standard threshold 0.5 works well with proper training
            predicted_observables = (torch.sigmoid(output) > 0.5).int().numpy()[0]
        
        if len(predicted_observables.shape) == 0:
            predicted_observables = predicted_observables.reshape(1)
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """
        Decode batch of syndromes with optimal threshold
        
        Args:
            syndromes: Batch of syndrome vectors
            
        Returns:
            predicted_observables: Batch of binary predictions
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            # Standard threshold
            predicted_observables = (torch.sigmoid(outputs) > 0.5).int().numpy()
        
        if len(predicted_observables.shape) == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        return predicted_observables.astype(np.uint8)