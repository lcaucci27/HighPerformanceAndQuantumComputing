import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os, glob
from utils import QECDataset, LLDModel

def train_lld():
    print("   [Step 2] Training LLD Models...")
    os.makedirs("checkpoints", exist_ok=True)
    
    for d in [3, 5, 7]:
        files = glob.glob(f"data/d{d}_*.npz")
        if not files: print(f"Dati non trovati per d={d}"); continue

        ds_temp = QECDataset(files[0])
        model = LLDModel(ds_temp.X.shape[1], ds_temp.y_corr.shape[1])
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss()
        
        best_val = float("inf"); patience = 0
        
        for ep in range(30): # Epochs ridotte per demo one-shot
            model.train()
            for f in files:
                loader = DataLoader(QECDataset(f, "train"), batch_size=64, shuffle=True)
                for x, _, y in loader:
                    opt.zero_grad()
                    loss = crit(model(x), y)
                    loss.backward(); opt.step()
            
            # Validation
            model.eval()
            val_loss = 0; count = 0
            with torch.no_grad():
                for f in files:
                    loader = DataLoader(QECDataset(f, "val"), batch_size=64)
                    for x, _, y in loader:
                        val_loss += crit(model(x), y).item(); count += 1
            
            avg = val_loss/count
            if avg < best_val:
                best_val = avg; patience = 0
                torch.save(model.state_dict(), f"checkpoints/lld_d{d}.pth")
            else:
                patience += 1
                if patience >= 3: break # Early stopping aggressivo
        print(f"      -> d={d} Trained. Best Val Loss: {best_val:.4f}")

if __name__ == "__main__": train_lld()
