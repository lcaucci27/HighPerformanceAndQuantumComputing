import os
import numpy as np
import stim
import pymatching
from tqdm import tqdm
from utils import get_stim_circuit

DISTANCES = [3, 5, 7]
PROBS = [0.001, 0.005, 0.01, 0.015, 0.02]
NUM_SAMPLES = 10000
DATA_DIR = "data"

def generate():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"   [Step 1] Generazione {NUM_SAMPLES} campioni per configurazione...")
    
    for d in DISTANCES:
        for p in PROBS:
            fname = f"{DATA_DIR}/d{d}_p{p}.npz"
            if os.path.exists(fname):
                print(f"Skipping {fname} (already exists)")
                continue 
            
            # Stim Generation
            c = get_stim_circuit(d, p)
            s = c.compile_detector_sampler()
            
            # FIX APPLICATO: Aggiunto separate_observables=True per ottenere la tupla (dets, obs)
            dets, obs = s.sample(NUM_SAMPLES, bit_packed=False, separate_observables=True)
            
            # MWPM Targets (Training LLD to imitate MWPM)
            m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
            corrs = m.decode_batch(dets)
            
            np.savez_compressed(fname,
                syndromes=np.packbits(dets.astype(np.uint8), axis=1),
                observables=obs.astype(np.uint8),
                corrections=np.packbits(corrs.astype(np.uint8), axis=1))

if __name__ == "__main__": generate()
