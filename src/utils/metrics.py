"""
Metrics calculation for decoder performance
Calculates LER, pseudothreshold, and slope
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def calculate_ler(actual_errors, corrections, actual_logicals):
    """
    Calculate Logical Error Rate
    
    Args:
        actual_errors: Binary array (num_samples, num_data_qubits) - not used, kept for API
        corrections: Binary array (num_samples, num_observables) - predicted logical flips
        actual_logicals: Binary array (num_samples, num_observables) - actual logical flips
    
    Returns:
        ler: Logical error rate (fraction of incorrect predictions)
    """
    # Compare predicted vs actual logical observables
    # corrections from decoders should be observable predictions
    
    # If corrections are data qubits, we need to handle differently
    # For now, assume they match logicals shape or need comparison
    
    if len(corrections.shape) == 1:
        corrections = corrections.reshape(-1, 1)
    if len(actual_logicals.shape) == 1:
        actual_logicals = actual_logicals.reshape(-1, 1)
    
    # Calculate errors - where prediction doesn't match actual
    errors = np.any(corrections != actual_logicals, axis=1)
    ler = np.mean(errors)
    
    return ler


def calculate_pseudothreshold(per_values, ler_values):
    """
    Calculate pseudothreshold (where LER = PER)
    
    Args:
        per_values: Physical error rates (array)
        ler_values: Logical error rates (array)
    
    Returns:
        pseudothreshold: PER value where LER ≈ PER
    """
    # Find intersection of LER curve with LER=PER line
    # Use interpolation in log space
    
    # Filter out zero or negative values
    valid_mask = (per_values > 0) & (ler_values > 0)
    per_values = per_values[valid_mask]
    ler_values = ler_values[valid_mask]
    
    if len(per_values) < 2:
        return np.median(per_values) if len(per_values) > 0 else 0.01
    
    log_per = np.log10(per_values)
    log_ler = np.log10(ler_values)
    
    # Find where log(LER) ≈ log(PER)
    diff = log_ler - log_per
    
    # Find sign change (crossing point)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) == 0:
        # No crossing found, return point where diff is closest to 0
        closest_idx = np.argmin(np.abs(diff))
        return per_values[closest_idx]
    
    # Take first crossing
    idx = sign_changes[0]
    
    # Linear interpolation between the two points
    x1, x2 = log_per[idx], log_per[idx + 1]
    y1, y2 = diff[idx], diff[idx + 1]
    
    # Find x where y = 0
    if y2 - y1 != 0:
        x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
        pseudothreshold = 10 ** x_cross
    else:
        pseudothreshold = per_values[idx]
    
    return pseudothreshold


def calculate_slope(per_values, ler_values):
    """
    Calculate decoder slope (rate of LER improvement with distance)
    
    Args:
        per_values: Physical error rates (array)
        ler_values: Logical error rates (array)
    
    Returns:
        slope: Slope in log-log space
    """
    # Filter out zero or negative values
    valid_mask = (per_values > 0) & (ler_values > 0) & (ler_values < per_values)
    
    if np.sum(valid_mask) < 2:
        return 1.0  # Default slope
    
    # Fit line in log-log space
    log_per = np.log10(per_values[valid_mask])
    log_ler = np.log10(ler_values[valid_mask])
    
    # Linear fit in log space
    coeffs = np.polyfit(log_per, log_ler, 1)
    slope = coeffs[0]
    
    return slope


def fit_error_model(per_values, ler_values):
    """
    Fit error model to data using equation from paper:
    l = p_th * (p / p_th)^(s * (1 - c*p))
    
    Args:
        per_values: Physical error rates
        ler_values: Logical error rates
    
    Returns:
        params: dict with p_th, s, c
    """
    def model(p, p_th, s, c):
        return p_th * (p / p_th) ** (s * (1 - c * p))
    
    try:
        # Initial guess
        p0 = [0.05, 2.0, 1.0]
        
        # Fit
        params, _ = curve_fit(model, per_values, ler_values, p0=p0, maxfev=10000)
        
        return {
            'p_th': params[0],
            's': params[1],
            'c': params[2]
        }
    except:
        # If fit fails, return defaults
        return {
            'p_th': calculate_pseudothreshold(per_values, ler_values),
            's': calculate_slope(per_values, ler_values),
            'c': 1.0
        }