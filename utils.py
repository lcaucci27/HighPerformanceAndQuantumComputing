import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import stim

class QECDataset(Dataset):
    def __init__(self, filepath, mode='train'):
        with np.load(filepath) as data:
            # Unpack on demand or preload. Preloading is fine for this dataset size.
            self.syndromes = torch.tensor(np.unpackbits(data['syndromes'], axis=1), dtype=torch.float32)
            self.observables = torch.tensor(data['observables'], dtype=torch.float32)
            self.corrections = torch.tensor(np.unpackbits(data['corrections'], axis=1), dtype=torch.float32)

        n = len(self.syndromes)
        s1, s2 = int(0.8*n), int(0.9*n)
        
        if mode == 'train':
            self.X, self.y_obs, self.y_corr = self.syndromes[:s1], self.observables[:s1], self.corrections[:s1]
        elif mode == 'val':
            self.X, self.y_obs, self.y_corr = self.syndromes[s1:s2], self.observables[s1:s2], self.corrections[s1:s2]
        else:
            self.X, self.y_obs, self.y_corr = self.syndromes[s2:], self.observables[s2:], self.corrections[s2:]

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y_obs[i], self.y_corr[i]

class LLDModel(nn.Module):
    def __init__(self, i_size, o_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, o_size), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class HLDModel(nn.Module):
    def __init__(self, s_size, c_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_size + c_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, s, c): return self.net(torch.cat([s, c], dim=1))

def get_stim_circuit(d, p):
    # FIX APPLICATO: Rimossa 'after_measure_flip_probability' non supportata.
    # Aggiunta 'before_round_data_depolarization' per mantenere il noise model robusto.
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=d,
        distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p
    )
