"""
Metrics calculation for decoder performance
Calculates LER, pseudothreshold, and slope with robust algorithms
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
    
    Uses robust interpolation in log-space with multiple strategies:
    1. Direct interpolation if crossing exists
    2. Extrapolation if no crossing in range
    3. Fallback to closest point
    
    Args:
        per_values: Physical error rates (array)
        ler_values: Logical error rates (array)
    
    Returns:
        pseudothreshold: PER value where LER ≈ PER
    """
    # Ensure numpy arrays
    per_values = np.array(per_values, dtype=np.float64)
    ler_values = np.array(ler_values, dtype=np.float64)
    
    # Filter out invalid values
    valid_mask = (per_values > 0) & (ler_values > 0) & np.isfinite(per_values) & np.isfinite(ler_values)
    per_values = per_values[valid_mask]
    ler_values = ler_values[valid_mask]
    
    if len(per_values) < 2:
        return float(np.median(per_values)) if len(per_values) > 0 else 0.01
    
    # Sort by PER
    sort_idx = np.argsort(per_values)
    per_values = per_values[sort_idx]
    ler_values = ler_values[sort_idx]
    
    # Work in log space for better behavior
    log_per = np.log10(per_values)
    log_ler = np.log10(ler_values)
    
    # Difference function: log(LER) - log(PER)
    # We want to find where this equals zero (LER = PER)
    diff = log_ler - log_per
    
    # Strategy 1: Look for sign changes (crossings)
    sign_changes = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:  # Sign change detected
            sign_changes.append(i)
    
    if len(sign_changes) > 0:
        # Use the first crossing (most relevant)
        idx = sign_changes[0]
        
        # Linear interpolation between the two bracketing points
        x1, x2 = log_per[idx], log_per[idx + 1]
        y1, y2 = diff[idx], diff[idx + 1]
        
        # Solve for x where y = 0
        if abs(y2 - y1) > 1e-12:
            x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
            pth = 10 ** x_cross
        else:
            # Points are essentially identical, use midpoint
            pth = 10 ** ((x1 + x2) / 2)
        
        return float(pth)
    
    # Strategy 2: No crossing found, use interpolation/extrapolation
    # Create interpolation function
    try:
        # Use cubic interpolation if we have enough points
        if len(per_values) >= 4:
            f_interp = interp1d(log_per, diff, kind='cubic', fill_value='extrapolate')
        else:
            f_interp = interp1d(log_per, diff, kind='linear', fill_value='extrapolate')
        
        # Search in extended range
        search_min = log_per[0] - 0.5
        search_max = log_per[-1] + 0.5
        
        # Try to find root
        try:
            # Sample the function to find a bracket
            search_points = np.linspace(search_min, search_max, 50)
            search_vals = f_interp(search_points)
            
            # Look for sign changes in extended range
            for i in range(len(search_vals) - 1):
                if search_vals[i] * search_vals[i+1] < 0:
                    # Found a bracket, use Brent's method
                    log_pth = brentq(f_interp, search_points[i], search_points[i+1])
                    pth = 10 ** log_pth
                    
                    # Sanity check: should be in reasonable range
                    if 0.0001 < pth < 0.5:
                        return float(pth)
        except:
            pass
    except:
        pass
    
    # Strategy 3: Use closest point to LER = PER
    closest_idx = np.argmin(np.abs(diff))
    pth = per_values[closest_idx]
    
    # Strategy 4: Adjust based on trend
    if diff[0] > 0:
        # LER > PER at low rates, threshold is below our range
        # Extrapolate downward conservatively
        if len(per_values) >= 2:
            # Estimate slope at lower end
            slope = (diff[1] - diff[0]) / (log_per[1] - log_per[0])
            if slope > 0 and abs(slope) < 10:  # Reasonable slope
                # Extrapolate
                delta_log = -diff[0] / slope
                log_pth = log_per[0] + delta_log
                pth_extrap = 10 ** log_pth
                # Use if reasonable
                if 0.0001 < pth_extrap < per_values[0]:
                    pth = pth_extrap
                else:
                    pth = per_values[0] * 0.7  # Conservative factor
            else:
                pth = per_values[0] * 0.7
        else:
            pth = per_values[0] * 0.8
            
    elif diff[-1] < 0:
        # LER < PER at high rates, threshold is above our range
        # Extrapolate upward conservatively
        if len(per_values) >= 2:
            slope = (diff[-1] - diff[-2]) / (log_per[-1] - log_per[-2])
            if slope < 0 and abs(slope) < 10:
                delta_log = -diff[-1] / slope
                log_pth = log_per[-1] + delta_log
                pth_extrap = 10 ** log_pth
                if per_values[-1] < pth_extrap < 0.5:
                    pth = pth_extrap
                else:
                    pth = per_values[-1] * 1.3
            else:
                pth = per_values[-1] * 1.3
        else:
            pth = per_values[-1] * 1.2
    
    return float(pth)


def calculate_slope(per_values, ler_values):
    """
    Calculate decoder slope: rate of LER improvement with distance
    
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
    
    # Fit line in log-log space: log(LER) = slope * log(PER) + intercept
    log_per = np.log10(per_values[valid_mask])
    log_ler = np.log10(ler_values[valid_mask])
    
    # Use polyfit for robust linear regression
    coeffs = np.polyfit(log_per, log_ler, 1)
    slope = coeffs[0]
    
    return float(slope)


def fit_error_model(per_values, ler_values):
    """
    Fit error model to data using exponential relationship
    
    Model: LER = p_th * (PER / p_th)^(slope * (1 - correction * PER))
    
    Args:
        per_values: Physical error rates
        ler_values: Logical error rates
    
    Returns:
        params: dict with p_th, slope, correction
    """
    def model(p, p_th, s, c):
        """Error model with correction term"""
        return p_th * (p / p_th) ** (s * (1 - c * p))
    
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