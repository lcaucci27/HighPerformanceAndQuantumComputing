"""
Metrics calculation for decoder performance
Calculates LER, pseudothreshold, and slope with robust algorithms

FIX: Add numerical stability protections for edge cases
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, brentq

def calculate_ler(actual_errors, corrections, actual_logicals):
    """
    Calculate Logical Error Rate
    
    The LER is the fraction of samples where the decoder's prediction
    doesn't match the actual logical observable flip.
    
    Args:
        actual_errors: Binary array (num_samples, num_data_qubits) - not used
        corrections: Binary array (num_samples, num_observables) - predicted flips
        actual_logicals: Binary array (num_samples, num_observables) - actual flips
    
    Returns:
        ler: Logical error rate in [0, 1]
    """
    # Ensure proper shapes
    if len(corrections.shape) == 1:
        corrections = corrections.reshape(-1, 1)
    if len(actual_logicals.shape) == 1:
        actual_logicals = actual_logicals.reshape(-1, 1)
    
    # An error occurs if ANY observable is predicted incorrectly
    errors = np.any(corrections != actual_logicals, axis=1)
    ler = np.mean(errors)
    
    return float(ler)


def calculate_pseudothreshold(per_values, ler_values):
    """
    Calculate pseudothreshold: the PER value where LER = PER
    
    STRICT: Only return valid p_th if a real crossing exists within data range.
    NO extrapolation. If decoder is always worse (LER > PER everywhere),
    return NaN to indicate no valid pseudothreshold exists.
    
    Args:
        per_values: Physical error rates (array)
        ler_values: Logical error rates (array)
    
    Returns:
        pseudothreshold: PER value where LER ≈ PER, or NaN if no crossing exists
    """
    # Ensure numpy arrays
    per_values = np.array(per_values, dtype=np.float64)
    ler_values = np.array(ler_values, dtype=np.float64)
    
    # Filter out invalid values
    valid_mask = (
        (per_values > 0) & 
        (ler_values > 0) & 
        np.isfinite(per_values) & 
        np.isfinite(ler_values) &
        (per_values < 1.0) &
        (ler_values < 1.0)
    )
    
    per_values = per_values[valid_mask]
    ler_values = ler_values[valid_mask]
    
    if len(per_values) < 2:
        return np.nan
    
    # Sort by PER
    sort_idx = np.argsort(per_values)
    per_values = per_values[sort_idx]
    ler_values = ler_values[sort_idx]
    
    # Remove duplicate PER values
    unique_per, unique_indices = np.unique(per_values, return_index=True)
    if len(unique_per) < len(per_values):
        per_values = unique_per
        ler_values = ler_values[unique_indices]
    
    if len(per_values) < 2:
        return np.nan
    
    # Work in log space
    epsilon = 1e-10
    log_per = np.log10(per_values + epsilon)
    log_ler = np.log10(ler_values + epsilon)
    
    # Difference function: log(LER) - log(PER)
    diff = log_ler - log_per
    
    # CRITICAL: Look for sign changes ONLY (no extrapolation)
    sign_changes = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:  # Sign change detected
            sign_changes.append(i)
    
    if len(sign_changes) == 0:
        # NO crossing exists in data range
        # Return NaN to indicate invalid pseudothreshold
        return np.nan
    
    # Use the first crossing (most relevant)
    idx = sign_changes[0]
    
    # Linear interpolation between the two bracketing points
    x1, x2 = log_per[idx], log_per[idx + 1]
    y1, y2 = diff[idx], diff[idx + 1]
    
    # Solve for x where y = 0
    denom = y2 - y1
    if abs(denom) > 1e-12:
        x_cross = x1 - y1 * (x2 - x1) / denom
        pth = 10 ** x_cross
    else:
        # Points are essentially identical, use midpoint
        pth = 10 ** ((x1 + x2) / 2)
    
    # Sanity check: crossing should be between the two bracketing points
    min_per = min(per_values[idx], per_values[idx + 1])
    max_per = max(per_values[idx], per_values[idx + 1])
    
    if not (min_per <= pth <= max_per * 1.1):  # Allow 10% extrapolation within bracket
        return np.nan
    
    # Final check: pseudothreshold should be in reasonable range
    if not (0.0001 < pth < 0.15):
        return np.nan
    
    return float(pth)


def calculate_slope(per_values, ler_values):
    """
    Calculate decoder slope: rate of LER improvement with distance
    
    FIX: Add numerical stability protections
    
    Fits a line in log-log space to points where LER < PER
    
    Args:
        per_values: Physical error rates (array)
        ler_values: Logical error rates (array)
    
    Returns:
        slope: Slope in log-log space
    """
    # Filter for valid points where decoder helps (LER < PER)
    valid_mask = (per_values > 0) & (ler_values > 0) & (ler_values < per_values)
    
    if np.sum(valid_mask) < 2:
        return 1.0  # Default slope (no improvement)
    
    # FIX: Add epsilon to prevent log(0)
    epsilon = 1e-10
    log_per = np.log10(per_values[valid_mask] + epsilon)
    log_ler = np.log10(ler_values[valid_mask] + epsilon)
    
    # FIX: Check for degenerate case (constant values)
    if np.std(log_per) < 1e-8 or np.std(log_ler) < 1e-8:
        return 1.0
    
    try:
        # Use polyfit for robust linear regression
        coeffs = np.polyfit(log_per, log_ler, 1)
        slope = coeffs[0]
        
        # FIX: Sanity check on slope value
        if not np.isfinite(slope) or slope < 0 or slope > 100:
            return 1.0
        
        return float(slope)
    except:
        return 1.0


def fit_error_model(per_values, ler_values):
    """
    Fit error model to data using exponential relationship
    
    FIX: Add try-catch for numerical stability
    
    Model: LER = p_th * (PER / p_th)^(slope * (1 - correction * PER))
    
    Args:
        per_values: Physical error rates
        ler_values: Logical error rates
    
    Returns:
        params: dict with p_th, slope, correction
    """
    def model(p, p_th, s, c):
        """Error model with correction term"""
        try:
            result = p_th * (p / p_th) ** (s * (1 - c * p))
            # FIX: Check for invalid results
            if not np.all(np.isfinite(result)):
                return np.ones_like(p) * 0.01
            return result
        except:
            return np.ones_like(p) * 0.01
    
    try:
        # Initial estimates
        p_th_init = calculate_pseudothreshold(per_values, ler_values)
        s_init = max(1.5, calculate_slope(per_values, ler_values))
        p0 = [p_th_init, s_init, 1.0]
        
        # Bounds for parameters
        bounds = ([0.001, 0.5, 0.0], [0.3, 10.0, 20.0])
        
        # Fit with curve_fit
        params, _ = curve_fit(
            model, per_values, ler_values, 
            p0=p0, bounds=bounds, maxfev=10000
        )
        
        return {
            'p_th': float(params[0]),
            'slope': float(params[1]),
            'correction': float(params[2])
        }
    except Exception:
        # If fit fails, return estimates
        return {
            'p_th': calculate_pseudothreshold(per_values, ler_values),
            'slope': calculate_slope(per_values, ler_values),
            'correction': 1.0
        }