import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os, glob
from utils import QECDataset, HLDModel

def train_hld():
    print("   [Step 3] Training HLD Models...")
    os.makedirs("checkpoints", exist_ok=True)
    
    for d in [3, 5, 7]:
        files = glob.glob(f"data/d{d}_*.npz")
        ds_temp = QECDataset(files[0])
        model = HLDModel(ds_temp.X.shape[1], ds_temp.y_corr.shape[1])
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss()
        
        best_val = float("inf"); patience = 0
        
        for ep in range(30):
            model.train()
            for f in files:
                loader = DataLoader(QECDataset(f, "train"), batch_size=64, shuffle=True)
                for s, obs, corr in loader:
                    opt.zero_grad()
                    loss = crit(model(s, corr), obs)
                    loss.backward(); opt.step()
            
            model.eval()
            val_loss = 0; count = 0
            with torch.no_grad():
                for f in files:
                    loader = DataLoader(QECDataset(f, "val"), batch_size=64)
                    for s, obs, corr in loader:
                        val_loss += crit(model(s, corr), obs).item(); count += 1
            
            avg = val_loss/count
            if avg < best_val:
                best_val = avg; patience = 0
                torch.save(model.state_dict(), f"checkpoints/hld_d{d}.pth")
            else:
                patience += 1
                if patience >= 3: break
        print(f"      -> d={d} Trained. Best Val Loss: {best_val:.4f}")

if __name__ == "__main__": train_hld()
