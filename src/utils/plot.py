"""
Plotting utilities for generating final graphs
Creates two main figures: per decoder and per distance
Cross-platform compatible with optimized file sizes
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from src.utils.metrics import calculate_pseudothreshold

# Global settings for high-quality, compact plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.figsize'] = (12, 3.5)

def plot_results(results, distances, output_dir):
    """
    Generate and save both main figures
    
    Args:
        results: Dict {decoder: {distance: {'per': [], 'ler': []}}}
        distances: List of code distances
        output_dir: Output directory path
    """
    # Figure 1: Per decoder comparison
    plot_per_decoder(results, distances, output_dir)
    
    # Figure 2: Per distance comparison
    plot_per_distance(results, distances, output_dir)


def plot_per_decoder(results, distances, output_dir):
    """
    Figure 1: Three subplots, one per decoder
    Shows how each decoder performs across distances
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle('Decoder Performance Comparison', 
                 fontsize=12, fontweight='bold', y=1.00)
    
    decoder_names = ['LLD', 'Baseline', 'HLD']
    decoder_keys = ['lld', 'baseline', 'hld']
    colors_per_dist = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for idx, (decoder_name, decoder_key) in enumerate(zip(decoder_names, decoder_keys)):
        ax = axes[idx]
        
        # Plot each distance
        for d_idx, d in enumerate(distances):
            per = np.array(results[decoder_key][d]['per'])
            ler = np.array(results[decoder_key][d]['ler'])
            
            # Sort by PER
            sort_idx = np.argsort(per)
            per = per[sort_idx]
            ler = ler[sort_idx]
            
            # Plot LER vs PER curve
            ax.loglog(per, ler, 'o-', label=f'd={d}', 
                     color=colors_per_dist[d_idx], 
                     linewidth=1.5, markersize=4)
            
            # Calculate and mark pseudothreshold
            pth = calculate_pseudothreshold(per, ler)
            
            # Mark pseudothreshold on diagonal
            ax.plot(pth, pth, 'o', markersize=9, 
                   color=colors_per_dist[d_idx], 
                   markerfacecolor='white', 
                   markeredgewidth=2, zorder=10)
            
            # Annotate pseudothreshold value
            if decoder_key == 'lld':
                offset_x, offset_y = 1.15, 0.80
            elif decoder_key == 'baseline':
                offset_x, offset_y = 1.10, 0.85
            else:  # hld
                offset_x, offset_y = 1.18, 0.75
            
            ax.text(pth * offset_x, pth * offset_y, f'{pth:.3f}', 
                   fontsize=7.5, color=colors_per_dist[d_idx], 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.25', 
                            facecolor='white', 
                            edgecolor=colors_per_dist[d_idx], 
                            alpha=0.9, linewidth=1))
        
        # Add LER=PER diagonal reference line
        per_line = np.logspace(np.log10(0.002), np.log10(0.15), 50)
        ax.loglog(per_line, per_line, 'k--', 
                 linewidth=1, alpha=0.4, label='LER=PER')
        
        # Formatting
        ax.set_xlabel('Physical Error Rate', fontsize=9, fontweight='bold')
        ax.set_ylabel('Logical Error Rate', fontsize=9, fontweight='bold')
        ax.set_title(f'{decoder_name}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7.5, loc='lower right', 
                 framealpha=0.95, edgecolor='gray', fancybox=False)
        ax.grid(True, alpha=0.25, which='both', 
               linestyle=':', linewidth=0.5)
        ax.set_xlim([0.002, 0.15])
        ax.set_ylim([0.0001, 0.3])
        ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    
    # Save with optimization
    output_path = os.path.join(output_dir, 'figure_decoders.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                pil_kwargs={'optimize': True, 'quality': 85})
    plt.close()
    print(f"   ✓ Saved: figure_decoders.png")


def plot_per_distance(results, distances, output_dir):
    """
    Figure 2: Three subplots, one per distance
    Shows how decoders compare at each distance
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle('Distance Comparison Across Decoders', 
                 fontsize=12, fontweight='bold', y=1.00)
    
    decoder_names = ['Baseline', 'LLD', 'HLD']
    decoder_keys = ['baseline', 'lld', 'hld']
    colors = {
        'baseline': '#2ca02c',  # Green
        'lld': '#d62728',       # Red
        'hld': '#1f77b4'        # Blue
    }
    markers = {
        'baseline': 's',  # Square
        'lld': '^',       # Triangle
        'hld': 'o'        # Circle
    }
    
    for d_idx, d in enumerate(distances):
        ax = axes[d_idx]
        
        # Plot each decoder
        for decoder_name, decoder_key in zip(decoder_names, decoder_keys):
            per = np.array(results[decoder_key][d]['per'])
            ler = np.array(results[decoder_key][d]['ler'])
            
            # Sort by PER
            sort_idx = np.argsort(per)
            per = per[sort_idx]
            ler = ler[sort_idx]
            
            # Plot curve
            ax.loglog(per, ler, 
                     marker=markers[decoder_key], 
                     linestyle='-', 
                     label=decoder_name, 
                     color=colors[decoder_key], 
                     linewidth=1.8, 
                     markersize=5, 
                     markeredgewidth=0.3,
                     markeredgecolor='white')
            
            # Calculate and mark pseudothreshold
            pth = calculate_pseudothreshold(per, ler)
            
            ax.plot(pth, pth, 
                   marker=markers[decoder_key], 
                   markersize=11, 
                   color=colors[decoder_key], 
                   markerfacecolor='white', 
                   markeredgewidth=2, 
                   zorder=10)
            
            # Annotate pseudothreshold
            if decoder_key == 'baseline':
                offset_x, offset_y = 0.70, 1.35
            elif decoder_key == 'hld':
                offset_x, offset_y = 0.68, 1.40
            else:  # lld
                offset_x, offset_y = 1.30, 0.72
            
            ax.text(pth * offset_x, pth * offset_y, f'{pth:.3f}', 
                   fontsize=7.5, color=colors[decoder_key], 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.25', 
                            facecolor='white', 
                            edgecolor=colors[decoder_key], 
                            alpha=0.9, linewidth=1))
        
        # Add LER=PER diagonal
        per_line = np.logspace(np.log10(0.002), np.log10(0.15), 50)
        ax.loglog(per_line, per_line, 'k--', 
                 linewidth=1, alpha=0.4, label='LER=PER')
        
        # Formatting
        ax.set_xlabel('Physical Error Rate', fontsize=9, fontweight='bold')
        ax.set_ylabel('Logical Error Rate', fontsize=9, fontweight='bold')
        ax.set_title(f'd={d}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7.5, loc='lower right', 
                 framealpha=0.95, edgecolor='gray', fancybox=False)
        ax.grid(True, alpha=0.25, which='both', 
               linestyle=':', linewidth=0.5)
        ax.set_xlim([0.002, 0.15])
        ax.set_ylim([0.0001, 0.3])
        ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    
    # Save with optimization
    output_path = os.path.join(output_dir, 'figure_distances.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                pil_kwargs={'optimize': True, 'quality': 85})
    plt.close()
    print(f"   ✓ Saved: figure_distances.png")


def plot_training_progress(losses, output_dir, decoder_name):
    """
    Optional: Plot training loss over epochs
    
    Args:
        losses: List of loss values per epoch
        output_dir: Output directory
        decoder_name: Decoder name for filename
    """
    plt.figure(figsize=(6, 3))
    plt.plot(losses, linewidth=1.5, color='#1f77b4')
    plt.xlabel('Epoch', fontsize=9, fontweight='bold')
    plt.ylabel('Loss', fontsize=9, fontweight='bold')
    plt.title(f'{decoder_name} Training Progress', 
             fontsize=10, fontweight='bold')
    plt.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{decoder_name}_training.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                pil_kwargs={'optimize': True, 'quality': 85})
    plt.close()