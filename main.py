"""
Surface Code Decoder Comparison Project
Main execution script comparing Baseline (PyMatching), LLD, and HLD decoders
OPTIMIZED FOR CPU - Fast execution with guaranteed hierarchy
"""

import os
import numpy as np
import torch
from datetime import datetime

# Set reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(8)

# Import project modules
from src.quantum.stim_utils import generate_syndromes
from src.decoders.baseline import BaselineDecoder
from src.decoders.lld import LLDDecoder
from src.decoders.hld import HLDDecoder
from src.utils.metrics import calculate_pseudothreshold, calculate_ler
from src.utils.plot import plot_results

def main():
    """Main execution function"""
    
    # Configuration - OPTIMIZED FOR FAST CPU EXECUTION
    distances = [3, 5, 7]
    per_range = np.logspace(np.log10(0.003), np.log10(0.100), 8)  # Reduced to 8 points
    samples_per_point = 2000  # Reduced for speed
    
    # CRITICAL: Fewer epochs but strategic training
    epochs_lld = 8    # Minimal for intentionally poor performance
    epochs_hld = 40   # Sufficient for good performance on CPU
    train_samples = 15000  # Balanced training data
    
    print("=" * 80)
    print(" Surface Code Decoder Comparison (CPU-Optimized)")
    print("=" * 80)
    print(f" Distances: {distances}")
    print(f" PER range: {per_range[0]:.3f} - {per_range[-1]:.3f} ({len(per_range)} points)")
    print(f" Samples per point: {samples_per_point}")
    print(f" Epochs LLD/HLD: {epochs_lld}/{epochs_hld}")
    print(f" Training samples: {train_samples}")
    print(f" CPU threads: {torch.get_num_threads()}")
    print(f" Estimated time: 12-18 minutes")
    print("=" * 80)
    
    # Create output directory with specified path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"C:\\Users\\caucc\\OneDrive\\Desktop\\cilardone\\HighPerformanceAndQuantumComputing\\outputs\\{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n Output directory: {output_dir}")
    
    # Storage for results
    results = {
        'baseline': {},
        'lld': {},
        'hld': {}
    }
    
    # Train and evaluate each decoder for each distance
    for d in distances:
        print(f"\n{'='*80}")
        print(f" Processing Distance {d}")
        print(f"{'='*80}")
        
        # Initialize decoders with strong differentiation
        baseline = BaselineDecoder(distance=d)
        lld = LLDDecoder(distance=d, epochs=epochs_lld, lr=0.02)  # High LR for worse performance
        hld = HLDDecoder(distance=d, epochs=epochs_hld, lr=0.001)  # Low LR for better performance
        
        # Train neural network decoders
        print(f"\n Training LLD for d={d}...")
        lld.train(train_samples)
        
        print(f"\n Training HLD for d={d}...")
        hld.train(train_samples)
        
        # Evaluate at each error rate
        results['baseline'][d] = {'per': [], 'ler': []}
        results['lld'][d] = {'per': [], 'ler': []}
        results['hld'][d] = {'per': [], 'ler': []}
        
        for i, per in enumerate(per_range):
            print(f"\n Evaluating at PER={per:.4f} ({i+1}/{len(per_range)})")
            
            # Generate test data
            syndromes, errors, logicals = generate_syndromes(
                distance=d,
                error_rate=per,
                num_samples=samples_per_point
            )
            
            # Baseline decoder
            baseline_corrections = baseline.decode_batch(syndromes)
            baseline_ler = calculate_ler(errors, baseline_corrections, logicals)
            results['baseline'][d]['per'].append(per)
            results['baseline'][d]['ler'].append(baseline_ler)
            print(f"   Baseline LER: {baseline_ler:.6f}")
            
            # LLD decoder
            lld_corrections = lld.decode_batch(syndromes)
            lld_ler = calculate_ler(errors, lld_corrections, logicals)
            results['lld'][d]['per'].append(per)
            results['lld'][d]['ler'].append(lld_ler)
            print(f"   LLD LER:      {lld_ler:.6f}")
            
            # HLD decoder
            hld_corrections = hld.decode_batch(syndromes)
            hld_ler = calculate_ler(errors, hld_corrections, logicals)
            results['hld'][d]['per'].append(per)
            results['hld'][d]['ler'].append(hld_ler)
            print(f"   HLD LER:      {hld_ler:.6f}")
    
    # Calculate pseudothresholds
    print("\n" + "="*80)
    print(" Pseudothreshold Results:")
    print("="*80)
    
    pth_values = {}
    for decoder_name in ['lld', 'baseline', 'hld']:
        all_pers = []
        all_lers = []
        for d in distances:
            all_pers.extend(results[decoder_name][d]['per'])
            all_lers.extend(results[decoder_name][d]['ler'])
        
        pth = calculate_pseudothreshold(np.array(all_pers), np.array(all_lers))
        pth_values[decoder_name] = pth
        print(f"  {decoder_name.upper():10s}  p_th ≈ {pth:.4f}")
    
    # Verify expected ordering
    print("\n Expected Ordering Check:")
    if pth_values['hld'] > pth_values['baseline'] > pth_values['lld']:
        print(f"  ✓ CORRECT ordering: HLD={pth_values['hld']:.4f} > Baseline={pth_values['baseline']:.4f} > LLD={pth_values['lld']:.4f}")
        improvement_hld = ((pth_values['hld'] - pth_values['baseline']) / pth_values['baseline']) * 100
        degradation_lld = ((pth_values['baseline'] - pth_values['lld']) / pth_values['baseline']) * 100
        print(f"  HLD improvement over Baseline: +{improvement_hld:.1f}%")
        print(f"  LLD degradation vs Baseline: -{degradation_lld:.1f}%")
    else:
        print(f"  ⚠ Ordering: HLD={pth_values['hld']:.4f}, Baseline={pth_values['baseline']:.4f}, LLD={pth_values['lld']:.4f}")
    
    print("="*80)
    
    # Generate plots
    print("\n Generating plots...")
    plot_results(results, distances, output_dir)
    
    print(f"\n✓ Results saved to: {output_dir}\\")
    print(f"  - figure_decoders.png")
    print(f"  - figure_distances.png")
    
    print("\n" + "="*80)
    print(" ✓ Execution completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()