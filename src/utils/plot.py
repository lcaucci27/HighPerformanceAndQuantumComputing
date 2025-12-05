"""
Plotting utilities for generating final graphs
Creates two main figures: per decoder and per distance
"""

import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics import calculate_pseudothreshold

def plot_results(results, distances, output_dir):
    """
    Generate and save final plots
    
    Args:
        results: Dict with structure {decoder: {distance: {'per': [], 'ler': []}}}
        distances: List of distances
        output_dir: Directory to save plots
    """
    # Figure 1: Per decoder (3 subplots, one per decoder)
    plot_per_decoder(results, distances, output_dir)
    
    # Figure 2: Per distance (3 subplots, one per distance)
    plot_per_distance(results, distances, output_dir)


def plot_per_decoder(results, distances, output_dir):
    """Plot Figure 1: One subplot per decoder"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Decoder Performance Comparison', fontsize=16, fontweight='bold')
    
    decoder_names = ['LLD', 'Baseline', 'HLD']
    decoder_keys = ['lld', 'baseline', 'hld']
    colors_per_dist = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for d=3,5,7
    
    for idx, (decoder_name, decoder_key) in enumerate(zip(decoder_names, decoder_keys)):
        ax = axes[idx]
        
        for d_idx, d in enumerate(distances):
            per = np.array(results[decoder_key][d]['per'])
            ler = np.array(results[decoder_key][d]['ler'])
            
            # Sort by per
            sort_idx = np.argsort(per)
            per = per[sort_idx]
            ler = ler[sort_idx]
            
            # Plot curve
            ax.loglog(per, ler, 'o-', label=f'd={d}', 
                     color=colors_per_dist[d_idx], linewidth=2, markersize=5)
            
            # Calculate and annotate pseudothreshold
            pth = calculate_pseudothreshold(per, ler)
            
            # Plot pseudothreshold marker on LER=PER line
            ax.plot(pth, pth, 'o', markersize=10, 
                   color=colors_per_dist[d_idx], markerfacecolor='none', markeredgewidth=2)
            
            # Annotate pseudothreshold value
            offset_x = 1.15 if idx == 0 else 1.2
            offset_y = 0.85 if idx == 0 else 0.8
            ax.text(pth * offset_x, pth * offset_y, f'{pth:.3f}', 
                   fontsize=9, color=colors_per_dist[d_idx], fontweight='bold')
        
        # Plot LER=PER line
        per_line = np.logspace(np.log10(0.002), np.log10(0.15), 100)
        ax.loglog(per_line, per_line, 'k--', linewidth=1.5, alpha=0.5, label='LER=PER')
        
        ax.set_xlabel('Physical Error Rate', fontsize=12)
        ax.set_ylabel('Logical Error Rate', fontsize=12)
        ax.set_title(f'{decoder_name} Decoder', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0.002, 0.15])
        ax.set_ylim([0.0001, 0.3])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure_decoders.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: figure_decoders.png")


def plot_per_distance(results, distances, output_dir):
    """Plot Figure 2: One subplot per distance"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distance Comparison Across Decoders', fontsize=16, fontweight='bold')
    
    decoder_names = ['Baseline', 'LLD', 'HLD']
    decoder_keys = ['baseline', 'lld', 'hld']
    colors = {'baseline': '#2ca02c', 'lld': '#d62728', 'hld': '#1f77b4'}  # Green, Red, Blue
    markers = {'baseline': 's', 'lld': '^', 'hld': 'o'}
    
    for d_idx, d in enumerate(distances):
        ax = axes[d_idx]
        
        for decoder_name, decoder_key in zip(decoder_names, decoder_keys):
            per = np.array(results[decoder_key][d]['per'])
            ler = np.array(results[decoder_key][d]['ler'])
            
            # Sort by per
            sort_idx = np.argsort(per)
            per = per[sort_idx]
            ler = ler[sort_idx]
            
            # Plot curve
            ax.loglog(per, ler, marker=markers[decoder_key], linestyle='-', 
                     label=decoder_name, color=colors[decoder_key], 
                     linewidth=2, markersize=6)
            
            # Calculate and annotate pseudothreshold
            pth = calculate_pseudothreshold(per, ler)
            
            # Plot pseudothreshold marker on LER=PER line
            ax.plot(pth, pth, marker=markers[decoder_key], markersize=12, 
                   color=colors[decoder_key], markerfacecolor='none', 
                   markeredgewidth=2.5)
            
            # Annotate pseudothreshold value
            offset_x = 0.7 if decoder_key == 'hld' else 1.3
            offset_y = 1.3 if decoder_key == 'hld' else 0.75
            ax.text(pth * offset_x, pth * offset_y, f'{pth:.3f}', 
                   fontsize=9, color=colors[decoder_key], fontweight='bold')
        
        # Plot LER=PER line
        per_line = np.logspace(np.log10(0.002), np.log10(0.15), 100)
        ax.loglog(per_line, per_line, 'k--', linewidth=1.5, alpha=0.5, label='LER=PER')
        
        ax.set_xlabel('Physical Error Rate', fontsize=12)
        ax.set_ylabel('Logical Error Rate', fontsize=12)
        ax.set_title(f'Distance {d}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0.002, 0.15])
        ax.set_ylim([0.0001, 0.3])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure_distances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: figure_distances.png")


def plot_training_progress(losses, output_dir, decoder_name):
    """
    Plot training loss progression
    
    Args:
        losses: List of loss values
        output_dir: Output directory
        decoder_name: Name of decoder (for filename)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{decoder_name} Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/{decoder_name}_training.png', dpi=300, bbox_inches='tight')
    plt.close()