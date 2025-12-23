import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pymatching
from utils import QECDataset, LLDModel, HLDModel, get_stim_circuit

RESULTS_DIR = "results"
DISTANCES = [3, 5, 7]
PROBS = [0.001, 0.005, 0.01, 0.015, 0.02]

def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {m: {d: {"x": [], "y": []} for d in DISTANCES} for m in ["MWPM", "LLD", "HLD"]}

    print("   [Step 4] Valutazione e Benchmark...")

    for d in DISTANCES:
        # Load Models
        t_ds = QECDataset(f"data/d{d}_p0.001.npz", "test")
        lld = LLDModel(t_ds.X.shape[1], t_ds.y_corr.shape[1]); lld.eval()
        hld = HLDModel(t_ds.X.shape[1], t_ds.y_corr.shape[1]); hld.eval()
        try:
            lld.load_state_dict(torch.load(f"checkpoints/lld_d{d}.pth"))
            hld.load_state_dict(torch.load(f"checkpoints/hld_d{d}.pth"))
        except:
            print(f"Warning: Checkpoints mancanti per d={d}")
            continue

        for p in PROBS:
            loader = torch.utils.data.DataLoader(QECDataset(f"data/d{d}_p{p}.npz", "test"), batch_size=1000)
            tot=0; e_m=0; e_l=0; e_h=0
            
            with torch.no_grad():
                for synd, obs_gt, mwpm_corr in loader:
                    tot += synd.shape[0]
                    # MWPM Live Decode
                    circ = get_stim_circuit(d, p)
                    matcher = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
                    pred_obs = matcher.decode_batch(synd.numpy().astype(np.uint8), bit_packed_predictions=False, bit_packed_shots=False)
                    pred_obs = pred_obs.reshape(-1, 1)
                    
                    e_m += (pred_obs != obs_gt.numpy()).any(axis=1).sum()
                    
                    # LLD
                    lld_bits = (lld(synd) > 0.5).float()
                    e_l += (lld_bits != mwpm_corr).any(dim=1).sum().item()
                    
                    # HLD (Correction on top of MWPM)
                    hld_dec = (hld(synd, mwpm_corr) > 0.5).float().numpy()
                    final = np.logical_xor(pred_obs, hld_dec)
                    e_h += (final != obs_gt.numpy()).any(axis=1).sum()

            for m, err in zip(["MWPM", "LLD", "HLD"], [e_m, e_l, e_h]):
                results[m][d]["x"].append(p)
                results[m][d]["y"].append(err/tot)

    # Plotting
    print("      -> Generazione Grafici...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, m in enumerate(["MWPM", "LLD", "HLD"]):
        for d in DISTANCES: axs[i].plot(results[m][d]["x"], results[m][d]["y"], "o-", label=f"d={d}")
        axs[i].plot([0.001, 0.02], [0.001, 0.02], "k--"); axs[i].set_title(m)
        axs[i].set_yscale("log"); axs[i].set_xscale("log"); axs[i].legend()
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/decoder_comparison.png", dpi=150)
    print("   [Step 4] Completato.")

if __name__ == "__main__": evaluate()
