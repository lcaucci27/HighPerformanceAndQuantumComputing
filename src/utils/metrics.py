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
    
    if len(corrections.shape) == 1:
        corrections = corrections.reshape(-1, 1)
    if len(actual_logicals.shape) == 1:
        actual_logicals = actual_logicals.reshape(-1, 1)
    
    # Calculate errors - where prediction doesn't match actual
    # For multiple observables, an error occurs if ANY observable is wrong
    errors = np.any(corrections != actual_logicals, axis=1)
    ler = np.mean(errors)
    
    return float(ler)


def calculate_pseudothreshold(per_values, ler_values):
    """
    Calculate pseudothreshold (where LER = PER)
    
    Args:
        per_values: Physical error rates (array)
        ler_values: Logical error rates (array)
    
    Returns:
        pseudothreshold: PER value where LER ≈ PER
    """
    # Ensure arrays
    per_values = np.array(per_values)
    ler_values = np.array(ler_values)
    
    # Filter out zero or negative values
    valid_mask = (per_values > 0) & (ler_values > 0)
    per_values = per_values[valid_mask]
    ler_values = ler_values[valid_mask]
    
    if len(per_values) < 2:
        return float(np.median(per_values)) if len(per_values) > 0 else 0.01
    
    # Sort by per_values
    sort_idx = np.argsort(per_values)
    per_values = per_values[sort_idx]
    ler_values = ler_values[sort_idx]
    
    # Work in log space for better interpolation
    log_per = np.log10(per_values)
    log_ler = np.log10(ler_values)
    
    # Find where log(LER) ≈ log(PER), i.e., where LER = PER
    diff = log_ler - log_per
    
    # Check for sign changes (crossing points)
    sign_changes = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:  # Sign change
            sign_changes.append(i)
    
    if len(sign_changes) == 0:
        # No crossing found, find point where diff is closest to 0
        closest_idx = np.argmin(np.abs(diff))
        
        # If all points are above or below, extrapolate carefully
        if diff[0] > 0:  # LER > PER everywhere, threshold is below range
            # Use linear extrapolation in log space
            if len(per_values) >= 2:
                slope = (log_ler[1] - log_ler[0]) / (log_per[1] - log_per[0])
                intercept = log_ler[0] - slope * log_per[0]
                # Solve slope * x + intercept = x for x (where log_ler = log_per)
                if slope != 1:
                    log_pth = -intercept / (slope - 1)
                    pth = 10 ** log_pth
                    # Clamp to reasonable range
                    pth = max(per_values[0] * 0.5, min(pth, per_values[0] * 0.9))
                else:
                    pth = per_values[0] * 0.8
            else:
                pth = per_values[0] * 0.8
        elif diff[-1] < 0:  # LER < PER everywhere, threshold is above range
            # Use linear extrapolation
            if len(per_values) >= 2:
                slope = (log_ler[-1] - log_ler[-2]) / (log_per[-1] - log_per[-2])
                intercept = log_ler[-1] - slope * log_per[-1]
                if slope != 1:
                    log_pth = -intercept / (slope - 1)
                    pth = 10 ** log_pth
                    pth = max(per_values[-1] * 1.1, min(pth, per_values[-1] * 1.5))
                else:
                    pth = per_values[-1] * 1.2
            else:
                pth = per_values[-1] * 1.2
        else:
            pth = per_values[closest_idx]
        
        return float(pth)
    
    # Take first crossing
    idx = sign_changes[0]
    
    # Linear interpolation between the two points
    x1, x2 = log_per[idx], log_per[idx + 1]
    y1, y2 = diff[idx], diff[idx + 1]
    
    # Find x where y = 0 (where log_ler - log_per = 0)
    if abs(y2 - y1) > 1e-10:
        x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
        pseudothreshold = 10 ** x_cross
    else:
        # Points are too close, take midpoint
        pseudothreshold = 10 ** ((x1 + x2) / 2)
    
    return float(pseudothreshold)


def calculate_slope(per_values, ler_values):
    """
    Calculate decoder slope (rate of LER improvement with distance)
    
    Args:
        per_values: Physical error rates (array)
        ler_values: Logical error rates (array)
    
    Returns:
        slope: Slope in log-log space
    """
    # Filter out zero or negative values and points where LER >= PER
    valid_mask = (per_values > 0) & (ler_values > 0) & (ler_values < per_values)
    
    if np.sum(valid_mask) < 2:
        return 1.0  # Default slope
    
    # Fit line in log-log space
    log_per = np.log10(per_values[valid_mask])
    log_ler = np.log10(ler_values[valid_mask])
    
    # Linear fit in log space: log(LER) = slope * log(PER) + intercept
    coeffs = np.polyfit(log_per, log_ler, 1)
    slope = coeffs[0]
    
    return float(slope)


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
        p_th_init = calculate_pseudothreshold(per_values, ler_values)
        s_init = calculate_slope(per_values, ler_values)
        p0 = [p_th_init, max(1.5, s_init), 1.0]
        
        # Fit with bounds
        bounds = ([0.001, 0.5, 0.0], [0.2, 10.0, 10.0])
        
        params, _ = curve_fit(
            model, per_values, ler_values, 
            p0=p0, bounds=bounds, maxfev=10000
        )
        
        return {
            'p_th': float(params[0]),
            's': float(params[1]),
            'c': float(params[2])
        }
    except Exception as e:
        # If fit fails, return defaults
        return {
            'p_th': calculate_pseudothreshold(per_values, ler_values),
            's': calculate_slope(per_values, ler_values),
            'c': 1.0
        }