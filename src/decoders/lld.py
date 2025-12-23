"""
Low-Level Decoder (LLD)
Neural network that maps syndromes directly to observable predictions
Intentionally simplified to perform worse than Baseline

FIX: Ensure model capacity scales with distance and can learn
distance-dependent patterns, but remains weak overall.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes

class LLDDecoder:
    """Low-Level Decoder - direct syndrome to observable mapping (naive)"""
    
    def __init__(self, distance, epochs=8, lr=0.01, hidden_size_factor=6):
        """
        Initialize LLD with simple direct prediction
        
        Args:
            distance: Code distance
            epochs: Number of training epochs (few)
            lr: Learning rate
            hidden_size_factor: Divisor for hidden size
        """
        self.distance = distance
        self.epochs = epochs
        self.lr = lr * 1.5  # FIX: Increase learning rate slightly for better learning
        self.hidden_size_factor = hidden_size_factor
        
        self.num_data_qubits = distance * distance
        
        # Get dimensions from Stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001
        )
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        
        # FIX: Network size MUST scale significantly with distance
        # Use 1.5x detectors for better capacity while staying "weak"
        hidden_size = max(96, int(self.num_detectors * 1.5))
        
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.15  # Moderate dropout
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, num_samples):
        """
        Train LLD on direct syndrome → observable mapping
        
        FIX: Use broader training range but fewer epochs to maintain weakness
        while allowing distance scaling
        """
        print(f"   Training LLD: {self.epochs} epochs, LR={self.lr}")
        
        # FIX: Train across multiple rates each epoch for better generalization
        # Include low rates where distance scaling matters most
        training_rates = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045]
        
        batch_size = 128
        samples_per_rate = num_samples // len(training_rates)
        
        for epoch in range(self.epochs):
            # Train on ALL rates each epoch (not cycling)
            all_X = []
            all_y = []
            
            for train_rate in training_rates:
                # Generate training data at this rate
                syndromes, _, logicals = generate_syndromes(
                    distance=self.distance,
                    error_rate=train_rate,
                    num_samples=samples_per_rate
                )
                
                all_X.append(syndromes)
                all_y.append(logicals)
            
            # Combine all training data
            X = torch.FloatTensor(np.vstack(all_X))
            y = torch.FloatTensor(np.vstack(all_y))
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # Shuffle
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            
            # Training loop
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                
                # Track accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.numel()
            
            avg_loss = epoch_loss / num_batches
            accuracy = correct / total
            
            if epoch == 0 or epoch == self.epochs - 1:
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
                      f"Acc: {accuracy:.3f}")
    
    def decode(self, syndrome):
        """Decode single syndrome using neural network"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            predicted = (torch.sigmoid(output) > 0.5).int().numpy()[0]
        
        if len(predicted.shape) == 0:
            predicted = predicted.reshape(1)
        
        # Ensure correct shape
        if len(predicted) < self.num_observables:
            padded = np.zeros(self.num_observables, dtype=np.uint8)
            padded[:len(predicted)] = predicted
            predicted = padded
        elif len(predicted) > self.num_observables:
            predicted = predicted[:self.num_observables]
        
        return predicted.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """Decode batch of syndromes using neural network"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            predicted = (torch.sigmoid(outputs) > 0.5).int().numpy()
        
        if len(predicted.shape) == 1:
            predicted = predicted.reshape(-1, 1)
        
        # Ensure correct shape
        num_samples = syndromes.shape[0]
        if predicted.shape[1] < self.num_observables:
            padded = np.zeros((num_samples, self.num_observables), dtype=np.uint8)
            padded[:, :predicted.shape[1]] = predicted
            predicted = padded
        elif predicted.shape[1] > self.num_observables:
            predicted = predicted[:, :self.num_observables]
        
        return predicted.astype(np.uint8)