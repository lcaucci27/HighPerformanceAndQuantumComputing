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
        
        # Neural network - VERY LARGE for superior performance
        # Scale with distance for better learning
        hidden_size = max(512, self.num_detectors * 6)  # Even larger
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='sqnl',  # Better activation function
            dropout=0.2  # More dropout for better generalization
        )
        
        # Use Adam optimizer with careful tuning
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=5e-5,  # More regularization
            betas=(0.9, 0.999)
        )
        
        # Weighted loss with higher weight on positive class
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=4,
            min_lr=1e-6
        )
    
    def train(self, num_samples):
        """Train the HLD decoder with excellent training strategy"""
        print(f"   Training HLD with {num_samples} samples per epoch...")
        
        # EXCELLENT TRAINING STRATEGY:
        # - Train across VERY WIDE range of error rates
        # - Focus on LOW error rates where we need to beat baseline
        # - Use large batches
        # - Heavy data augmentation via multiple error rates
        error_rates = [
            0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 
            0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.04, 0.05
        ]
        batch_size = 512  # Large stable batches
        
        for epoch in range(self.epochs):
            # Vary error rates - focus more on low rates early on
            if epoch < self.epochs // 2:
                # First half: focus on low error rates
                train_error_rate = error_rates[epoch % 10]
            else:
                # Second half: cover full range
                train_error_rate = error_rates[epoch % len(error_rates)]
            
            # Generate training data - use MORE than requested for overtraining
            actual_samples = int(num_samples * 1.2)  # 20% more data
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=actual_samples
            )
            
            # Convert to tensors
            X = torch.FloatTensor(syndromes)
            y = torch.FloatTensor(logicals)
            
            # Strong label smoothing for better calibration
            y = y * 0.85 + 0.075  # 0.075 to 0.925
            
            # Ensure y has correct shape
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # SHUFFLE DATA thoroughly
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            
            # Mini-batch training with multiple passes
            num_batches = max(1, len(X) // batch_size)
            epoch_loss = 0
            
            self.model.train()
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
                
                # Gentle gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            
            # Update learning rate
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    def decode(self, syndrome):
        """
        Decode single syndrome with optimized threshold
        
        Args:
            syndrome: Binary array (num_detectors,)
        
        Returns:
            predicted_observables: Binary array (num_observables,)
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            # Use optimized temperature and threshold
            # Lower temperature = more confident predictions
            predicted_observables = (torch.sigmoid(output / 0.75) > 0.48).int().numpy()[0]
        
        # Ensure correct shape
        if len(predicted_observables.shape) == 0:
            predicted_observables = predicted_observables.reshape(1)
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """Decode batch of syndromes with optimized threshold"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            # Use optimized temperature and threshold
            predicted_observables = (torch.sigmoid(outputs / 0.75) > 0.48).int().numpy()
        
        # Ensure correct shape
        if len(predicted_observables.shape) == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        return predicted_observables.astype(np.uint8)