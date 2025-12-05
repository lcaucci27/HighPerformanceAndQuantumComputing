"""
Stim utilities for generating syndromes and random errors
Uses Stim library to simulate surface code with depolarizing noise
"""

import numpy as np
import stim

def generate_syndromes(distance, error_rate, num_samples):
    """
    Generate syndrome measurements for surface code using Stim
    
    Args:
        distance: Surface code distance (3, 5, 7, or 9)
        error_rate: Physical error rate (probability per qubit)
        num_samples: Number of syndrome samples to generate
    
    Returns:
        syndromes: Binary array (num_samples, num_detectors)
        errors: Binary array (num_samples, num_data_qubits) 
        logicals: Binary array (num_samples, num_observables) - actual logical observables
    """
    
    # Generate surface code circuit with noise
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=distance,
        distance=distance,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate
    )
    
    # Create sampler
    sampler = circuit.compile_detector_sampler()
    
    # Sample detection events and observables
    detection_events, observable_flips = sampler.sample(
        shots=num_samples,
        separate_observables=True
    )
    
    # Convert to numpy arrays
    syndromes = detection_events.astype(np.uint8)
    logicals = observable_flips.astype(np.uint8)
    
    # Ensure logicals has correct shape (num_samples, num_observables)
    if len(logicals.shape) == 1:
        logicals = logicals.reshape(-1, 1)
    
    # For errors, we don't actually need them for our decoders
    # But keep the interface consistent
    num_data_qubits = distance * distance
    errors = np.zeros((num_samples, num_data_qubits), dtype=np.uint8)
    
    return syndromes, errors, logicals


def get_surface_code_params(distance):
    """
    Get parameters for surface code of given distance
    
    Args:
        distance: Code distance
    
    Returns:
        dict with num_data_qubits, num_detectors
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=distance,
        distance=distance,
        after_clifford_depolarization=0.001
    )
    
    num_data_qubits = distance * distance
    num_detectors = circuit.num_detectors
    num_observables = circuit.num_observables
    
    return {
        'num_data_qubits': num_data_qubits,
        'num_detectors': num_detectors,
        'num_observables': num_observables
    }


def apply_depolarizing_error(state, error_rate):
    """
    Apply depolarizing error to quantum state
    
    Args:
        state: Binary array representing qubit states
        error_rate: Probability of error per qubit
    
    Returns:
        New state with errors applied
    """
    errors = np.random.random(state.shape) < error_rate
    return (state + errors) % 2