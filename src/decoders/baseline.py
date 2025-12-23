"""
Baseline decoder using PyMatching (Minimum Weight Perfect Matching)
Classic decoder based on MWPM algorithm
"""

import numpy as np
import stim
import pymatching

class BaselineDecoder:
    """Baseline decoder using PyMatching for MWPM decoding"""
    
    def __init__(self, distance):
        """
        Initialize baseline decoder
        
        Args:
            distance: Surface code distance
        """
        self.distance = distance
        self.num_data_qubits = distance * distance
        
        # Create matching object from surface code circuit
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=0.001  # Nominal error rate for graph
        )
        
        # Get detector error model
        dem = circuit.detector_error_model(decompose_errors=True)
        
        # Create PyMatching decoder from DEM
        self.matcher = pymatching.Matching.from_detector_error_model(dem)
        
        self.num_observables = circuit.num_observables
    
    def decode(self, syndrome):
        """
        Decode a single syndrome using MWPM
        
        Args:
            syndrome: Binary array (num_detectors,)
        
        Returns:
            predicted_observables: Binary array (num_observables,)
        """
        # PyMatching returns predicted observable flips
        predicted_observables = self.matcher.decode(syndrome)
        
        # Ensure correct shape
        if len(predicted_observables) < self.num_observables:
            padded = np.zeros(self.num_observables, dtype=np.uint8)
            padded[:len(predicted_observables)] = predicted_observables
            predicted_observables = padded
        elif len(predicted_observables) > self.num_observables:
            predicted_observables = predicted_observables[:self.num_observables]
        
        return predicted_observables.astype(np.uint8)
    
    def decode_batch(self, syndromes):
        """
        Decode a batch of syndromes
        
        Args:
            syndromes: Binary array (num_samples, num_detectors)
        
        Returns:
            predicted_observables: Binary array (num_samples, num_observables)
        """
        # Use batch decoding
        predicted_observables = self.matcher.decode_batch(syndromes)
        
        # Ensure correct shape
        num_samples = syndromes.shape[0]
        if predicted_observables.shape[1] < self.num_observables:
            padded = np.zeros((num_samples, self.num_observables), dtype=np.uint8)
            padded[:, :predicted_observables.shape[1]] = predicted_observables
            predicted_observables = padded
        elif predicted_observables.shape[1] > self.num_observables:
            predicted_observables = predicted_observables[:, :self.num_observables]
        
        return predicted_observables.astype(np.uint8)