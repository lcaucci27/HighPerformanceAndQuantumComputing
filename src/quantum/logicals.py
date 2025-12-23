"""
Logical error calculation and classification
Determines if corrections match actual errors up to stabilizer operations
"""

import numpy as np

def calculate_logical_error(actual_errors, corrections, distance):
    """
    Calculate if there is a logical error after correction
    
    Args:
        actual_errors: Binary array (num_samples, num_data_qubits)
        corrections: Binary array (num_samples, num_data_qubits)
        distance: Code distance
    
    Returns:
        logical_errors: Binary array (num_samples,) indicating logical error
    """
    # XOR actual errors with corrections to get residual error
    residual = (actual_errors + corrections) % 2
    
    # Check if residual forms a logical operator
    # For rotated surface code, logical Z is a horizontal chain
    # Logical X is a vertical chain
    logical_errors = check_logical_chain(residual, distance)
    
    return logical_errors


def check_logical_chain(residual, distance):
    """
    Check if residual error forms a logical operator chain
    
    Args:
        residual: Binary array (num_samples, num_data_qubits)
        distance: Code distance
    
    Returns:
        is_logical: Binary array (num_samples,) 
    """
    num_samples = residual.shape[0]
    is_logical = np.zeros(num_samples, dtype=np.uint8)
    
    # Reshape to 2D lattice
    for i in range(num_samples):
        lattice = residual[i].reshape(distance, distance)
        
        # Check horizontal chain (logical Z)
        horizontal_parity = np.sum(lattice, axis=1) % 2
        has_horizontal = np.any(horizontal_parity == 1)
        
        # Check vertical chain (logical X)
        vertical_parity = np.sum(lattice, axis=0) % 2
        has_vertical = np.any(vertical_parity == 1)
        
        # Logical error if odd number of chains
        is_logical[i] = (has_horizontal or has_vertical).astype(np.uint8)
    
    return is_logical


def classify_logical_error(residual, distance):
    """
    Classify logical error as I, X, Y, or Z
    
    Args:
        residual: Binary array (num_data_qubits,)
        distance: Code distance
    
    Returns:
        error_class: 0=I, 1=X, 2=Y, 3=Z
    """
    lattice = residual.reshape(distance, distance)
    
    # Check for logical X (vertical chain)
    vertical_parity = np.sum(lattice, axis=0) % 2
    has_x = np.any(vertical_parity == 1)
    
    # Check for logical Z (horizontal chain)
    horizontal_parity = np.sum(lattice, axis=1) % 2
    has_z = np.any(horizontal_parity == 1)
    
    if has_x and has_z:
        return 2  # Y error
    elif has_x:
        return 1  # X error
    elif has_z:
        return 3  # Z error
    else:
        return 0  # I (no error)


def compute_parity(errors, logicals):
    """
    Compute parity of errors with respect to logical operators
    
    Args:
        errors: Binary array (num_data_qubits,)
        logicals: Binary array (num_data_qubits,) - logical operator
    
    Returns:
        parity: 0 or 1
    """
    return np.dot(errors, logicals) % 2