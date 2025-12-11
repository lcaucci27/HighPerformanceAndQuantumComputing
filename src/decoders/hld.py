"""
High-Level Decoder (HLD)
Advanced neural network for logical error classification
COMPLETELY REDESIGNED - Fixed training issues
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
    
    def __init__(self, distance, epochs=50, lr=0.002, hidden_size_factor=3):
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
        
        # LARGE architecture for superior capacity
        hidden_size = max(128, self.num_detectors * self.hidden_size_factor)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.15
        )
        
        # Adam optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=1e-5
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler for adaptive training
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.6, 
            patience=7,
            min_lr=1e-6
        )
    
    def train(self, num_samples):
        """
        Train HLD with FIXED strategy - actually learns!
        Key fixes:
        - Train on LOW error rates where class imbalance isn't extreme
        - Use multiple error rates per epoch for diversity
        - Proper batch normalization and convergence
        """
        print(f"   Training HLD: {self.epochs} epochs, LR={self.lr}")
        
        # CRITICAL FIX: Focus on LOW error rates where observables flip 5-20% of time
        # This is where the decoder should excel and learn meaningful patterns
        strategic_error_rates = [
            0.008, 0.010, 0.012, 0.015, 0.018,  # Low rates - good for learning
            0.020, 0.025, 0.030, 0.035, 0.040   # Medium rates - for generalization
        ]
        
        batch_size = 256
        samples_per_epoch = num_samples
        
        for epoch in range(self.epochs):
            # Progressive curriculum
            if epoch < 10:
                # Early: focus on low rates (easier to learn)
                active_rates = strategic_error_rates[0:5]
            elif epoch < 25:
                # Middle: expand range
                active_rates = strategic_error_rates[0:8]
            else:
                # Late: full range
                active_rates = strategic_error_rates
            
            # Generate training data from MULTIPLE error rates
            all_syndromes = []
            all_logicals = []
            
            samples_per_rate = samples_per_epoch // len(active_rates)
            
            for train_rate in active_rates:
                syndromes, _, logicals = generate_syndromes(
                    distance=self.distance,
                    error_rate=train_rate,
                    num_samples=samples_per_rate
                )
                all_syndromes.append(syndromes)
                all_logicals.append(logicals)
            
            # Combine all data
            X = torch.FloatTensor(np.vstack(all_syndromes))
            y = torch.FloatTensor(np.vstack(all_logicals))
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # Shuffle thoroughly
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            
            # Track metrics
            positive_rate = y.mean().item()
            
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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                
                # Track accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.numel()
            
            avg_loss = epoch_loss / num_batches
            accuracy = correct / total
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
                      f"Acc: {accuracy:.3f}, LR: {current_lr:.6f}, PosRate: {positive_rate:.2%}")
    
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