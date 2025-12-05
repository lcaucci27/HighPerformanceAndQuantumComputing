"""
High-Level Decoder (HLD)
Pure Error Decoder + Neural Network for logical error classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import stim
from src.models.mlp import MLP
from src.quantum.stim_utils import generate_syndromes
from src.quantum.logicals import classify_logical_error

class PureErrorDecoder:
    """Pure Error Decoder - generates valid data qubit configuration from syndrome"""
    
    def __init__(self, distance):
        self.distance = distance
        self.num_data_qubits = distance * distance
        
        # Get actual number of detectors from Stim
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001
        )
        self.num_detectors = circuit.num_detectors
    
    def decode(self, syndrome):
        """
        Decode syndrome to pure error (valid data qubit configuration)
        
        Args:
            syndrome: Binary array (num_detectors,)
        
        Returns:
            pure_error: Binary array (num_data_qubits,)
        """
        d = self.distance
        pure_error = np.zeros(self.num_data_qubits, dtype=np.uint8)
        
        # Simple greedy algorithm: map syndrome errors to data qubits
        # Each detector corresponds roughly to a region of data qubits
        num_regions = min(len(syndrome), self.num_data_qubits)
        
        for i in range(num_regions):
            if i < len(syndrome) and syndrome[i] == 1:
                # Map detector to corresponding data qubit
                data_idx = i % self.num_data_qubits
                pure_error[data_idx] = 1
        
        return pure_error
    
    def decode_batch(self, syndromes):
        """Decode batch of syndromes"""
        return np.array([self.decode(s) for s in syndromes])


class HLDDecoder:
    """High-Level Decoder: PED + NN for logical error classification"""
    
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
        
        # Pure Error Decoder
        self.ped = PureErrorDecoder(distance)
        
        # Neural network for observable prediction
        # Input: syndrome, Output: observable flips
        hidden_size = max(128, self.num_detectors * 2)
        self.model = MLP(
            input_size=self.num_detectors,
            hidden_sizes=[hidden_size, hidden_size // 2],
            output_size=self.num_observables,
            activation='sqnl'
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, num_samples):
        """Train the HLD decoder"""
        print(f"   Training HLD with {num_samples} samples per epoch...")
        
        # Training error rates - sweep for robustness
        error_rates = [0.01, 0.03, 0.05, 0.07]
        batch_size = 512
        
        for epoch in range(self.epochs):
            # Use different error rates
            train_error_rate = error_rates[epoch % len(error_rates)]
            
            # Generate training data
            syndromes, _, logicals = generate_syndromes(
                distance=self.distance,
                error_rate=train_error_rate,
                num_samples=num_samples
            )
            
            # Convert to tensors
            X = torch.FloatTensor(syndromes)
            y = torch.FloatTensor(logicals).reshape(-1, self.num_observables)
            
            # Mini-batch training
            num_batches = len(X) // batch_size
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
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
            predicted_observables = (torch.sigmoid(output) > 0.5).int().numpy()[0]
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """Decode batch of syndromes"""
        # Predict observables with NN
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(syndromes)
            outputs = self.model(X)
            predicted_observables = (torch.sigmoid(outputs) > 0.5).int().numpy()
        
        return predicted_observables.astype(np.uint8)
    
    def apply_logical_correction(self, pure_error, log_class):
        """Apply logical operator based on classification"""
        correction = pure_error.copy()
        
        d = self.distance
        
        # Apply logical X (vertical chain) if class is X or Y
        if log_class in [1, 2]:  # X or Y
            for i in range(0, d):
                correction[i] = (correction[i] + 1) % 2
        
        # Apply logical Z (horizontal chain) if class is Z or Y
        if log_class in [2, 3]:  # Y or Z
            for i in range(0, self.num_data_qubits, d):
                correction[i] = (correction[i] + 1) % 2
        
        return correction