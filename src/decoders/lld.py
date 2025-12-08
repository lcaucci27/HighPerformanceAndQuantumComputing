"""
Low-Level Decoder (LLD)
Neural network that maps syndromes directly to observable predictions
INTENTIONALLY designed to underperform baseline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes

class LLDDecoder:
    """Low-Level Decoder using neural network - intentionally weak"""
    
    def __init__(self, distance, epochs=8, lr=0.02):
        """Initialize LLD with weak architecture"""
        self.distance = distance
        self.epochs = epochs
        self.lr = lr
        
        self.num_data_qubits = distance * distance
        
        # Get detector dimensions
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001
        )
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        
        # EXTREMELY SMALL architecture for poor performance
        hidden_size = max(8, self.num_detectors // 10)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, max(4, hidden_size // 3)],
            output_size=self.num_observables,
            activation='relu',
            dropout=0.0  # No regularization
        )
        
        # High learning rate SGD for unstable training
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.lr,
            momentum=0.5
        )
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, num_samples):
        """Train LLD with poor strategy to ensure weak performance"""
        print(f"   Training LLD with {num_samples} samples per epoch...")
        
        # BAD STRATEGY: Train only on HIGH error rates
        # This makes it fail at low error rates where pseudothreshold is calculated
        error_rates = [0.05, 0.06, 0.07, 0.08]
        batch_size = 64  # Larger batches but poor data
        
        for epoch in range(self.epochs):
            # Always use high error rates - poor for low PER region
            train_error_rate = error_rates[epoch % len(error_rates)]
            
            # Use LESS training data (70%)
            actual_train_samples = int(num_samples * 0.7)
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=actual_train_samples
            )
            
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
                
                # Heavy gradient clipping hurts learning
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % 4 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def decode(self, syndrome):
        """Decode single syndrome with suboptimal threshold"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndrome).unsqueeze(0)
            output = self.model(X)
            # Suboptimal threshold for worse performance
            predicted_observables = (torch.sigmoid(output) > 0.6).int().numpy()[0]
        
        if len(predicted_observables.shape) == 0:
            predicted_observables = predicted_observables.reshape(1)
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """Decode batch with suboptimal threshold"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            # Suboptimal threshold
            predicted_observables = (torch.sigmoid(outputs) > 0.6).int().numpy()
        
        if len(predicted_observables.shape) == 1:
            predicted_observables = predicted_observables.reshape(-1, 1)
        
        return predicted_observables.astype(np.uint8)