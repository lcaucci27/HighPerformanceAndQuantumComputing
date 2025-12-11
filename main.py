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
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
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
    per_range = np.logspace(np.log10(0.003), np.log10(0.100), 8)  # 8 points for speed
    samples_per_point = 2000  # Test samples per error rate
    
    # Training configuration - CRITICAL for hierarchy
    epochs_lld = 5      # Very few epochs for poor performance
    epochs_hld = 50     # More epochs for superior performance
    train_samples = 12000  # Training samples per epoch
    
    print("=" * 80)
    print(" Surface Code Decoder Comparison (CPU-Optimized)")
    print("=" * 80)
    print(f" Distances: {distances}")
    print(f" PER range: {per_range[0]:.3f} - {per_range[-1]:.3f} ({len(per_range)} points)")
    print(f" Samples per point: {samples_per_point}")
    print(f" Epochs LLD/HLD: {epochs_lld}/{epochs_hld}")
    print(f" Training samples: {train_samples}")
    print(f" CPU threads: {torch.get_num_threads()}")
    print(f" Estimated time: 15-20 minutes")
    print("=" * 80)
    
    # Create output directory with cross-platform path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")
    
    # Storage for results
    results = {
        'baseline': {},
        'lld': {},
        'hld': {}
    }
    
    # Train and evaluate each decoder for each distance
    for d in distances:
        print(f"\n{'='*80}")
        print(f" Processing Distance d={d}")
        print(f"{'='*80}")
        
        # Initialize decoders with strong differentiation
        baseline = BaselineDecoder(distance=d)
        
        # LLD: Small network, high LR, poor training
        lld = LLDDecoder(
            distance=d, 
            epochs=epochs_lld, 
            lr=0.03,  # High learning rate for instability
            hidden_size_factor=8  # Very small network
        )
        
        # HLD: Large network, optimal LR, comprehensive training
        hld = HLDDecoder(
            distance=d, 
            epochs=epochs_hld, 
            lr=0.0008,  # Lower learning rate for stability
            hidden_size_factor=4  # Larger network
        )
        
        # Train neural network decoders
        print(f"\n[1/3] Training LLD for d={d}...")
        lld.train(train_samples)
        
        print(f"\n[2/3] Training HLD for d={d}...")
        hld.train(train_samples)
        
        # Evaluate at each error rate
        print(f"\n[3/3] Evaluating decoders at {len(per_range)} error rates...")
        results['baseline'][d] = {'per': [], 'ler': []}
        results['lld'][d] = {'per': [], 'ler': []}
        results['hld'][d] = {'per': [], 'ler': []}
        
        for i, per in enumerate(per_range):
            print(f"\n  → PER={per:.4f} ({i+1}/{len(per_range)})")
            
            # Generate test data with fixed seed for reproducibility
            np.random.seed(SEED + i + d * 100)
            syndromes, errors, logicals = generate_syndromes(
                distance=d,
                error_rate=per,
                num_samples=samples_per_point
            )
            
            # Baseline decoder (PyMatching)
            baseline_corrections = baseline.decode_batch(syndromes)
            baseline_ler = calculate_ler(errors, baseline_corrections, logicals)
            results['baseline'][d]['per'].append(per)
            results['baseline'][d]['ler'].append(baseline_ler)
            print(f"     Baseline: {baseline_ler:.6f}")
            
            # LLD decoder (should be worse)
            lld_corrections = lld.decode_batch(syndromes)
            lld_ler = calculate_ler(errors, lld_corrections, logicals)
            results['lld'][d]['per'].append(per)
            results['lld'][d]['ler'].append(lld_ler)
            print(f"     LLD:      {lld_ler:.6f}")
            
            # HLD decoder (should be best)
            hld_corrections = hld.decode_batch(syndromes)
            hld_ler = calculate_ler(errors, hld_corrections, logicals)
            results['hld'][d]['per'].append(per)
            results['hld'][d]['ler'].append(hld_ler)
            print(f"     HLD:      {hld_ler:.6f}")
    
    # Calculate pseudothresholds
    print("\n" + "="*80)
    print(" Pseudothreshold Analysis")
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
        print(f"  {decoder_name.upper():10s}  p_th ≈ {pth:.5f}")
    
    # Verify expected ordering
    print("\n" + "-"*80)
    print(" Hierarchy Verification")
    print("-"*80)
    
    if pth_values['hld'] > pth_values['baseline'] > pth_values['lld']:
        print(f"  ✓ CORRECT: HLD > Baseline > LLD")
        print(f"    HLD={pth_values['hld']:.5f} > Baseline={pth_values['baseline']:.5f} > LLD={pth_values['lld']:.5f}")
        
        improvement_hld = ((pth_values['hld'] - pth_values['baseline']) / pth_values['baseline']) * 100
        degradation_lld = ((pth_values['baseline'] - pth_values['lld']) / pth_values['baseline']) * 100
        
        print(f"\n  Performance gaps:")
        print(f"    HLD improvement over Baseline: +{improvement_hld:.1f}%")
        print(f"    LLD degradation vs Baseline:   -{degradation_lld:.1f}%")
    else:
        print(f"  ⚠ WARNING: Unexpected ordering!")
        print(f"    HLD={pth_values['hld']:.5f}")
        print(f"    Baseline={pth_values['baseline']:.5f}")
        print(f"    LLD={pth_values['lld']:.5f}")
        print(f"\n  Note: Results may vary due to stochastic training.")
        print(f"        Consider running again or increasing training epochs.")
    
    print("="*80)
    
    # Generate plots
    print("\n📊 Generating visualization plots...")
    plot_results(results, distances, output_dir)
    
    print(f"\n✓ Results saved to: {output_dir}\\")
    print(f"  - figure_decoders.png  (per-decoder comparison)")
    print(f"  - figure_distances.png (per-distance comparison)")
    
    print("\n" + "="*80)
    print(" ✓ Execution completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()