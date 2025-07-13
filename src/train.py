from pathlib import Path
from typing  import Tuple, List
import math, time, json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from timeseries_datamodule import merge_calendars, build_loaders
from model      import HierForecastNet, DEFAULT_HIDDEN_SIZE

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BATCH    = 64
EPOCHS   = 40
LR       = 2e-4
HID      = DEFAULT_HIDDEN_SIZE
W_DAY, W_WK, W_MTH = 0.1, 0.3, 0.6
SEED     = 42
torch.manual_seed(SEED); np.random.seed(SEED)


class HierWRMSSE(nn.Module):
    def __init__(self, w_d=W_DAY, w_w=W_WK, w_m=W_MTH):
        super().__init__(); self.wd, self.ww, self.wm = w_d, w_w, w_m
    def _rmsse(self, y_hat, y, ins):
        num   = torch.mean((y_hat - y)**2, dim=-1)
        denom = torch.mean((ins[:,1:] - ins[:,:-1])**2, dim=-1)
        return torch.sqrt(num / (denom + 1e-8))
    def forward(self, yd_hat, yw_hat, ym_hat,
                      yd,      yw,      ym,
                      ins_d,   ins_w,   ins_m):
        l_d = self._rmsse(yd_hat, yd, ins_d)
        l_w = self._rmsse(yw_hat, yw, ins_w)
        l_m = self._rmsse(ym_hat, ym, ins_m)
        return (self.wd*l_d + self.ww*l_w + self.wm*l_m).mean()


@torch.no_grad()
def bootstrap_fifos(model: HierForecastNet,
                    lookback: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    lookback: (B, 365, F) → produces
        week_fifo  : (B, 7, hidden_dim)
        month_fifo : (B,12, hidden_dim)
    """
    model.eval()
    B, _, _ = lookback.shape
    week_tokens: List[torch.Tensor] = []

    # Generate weekly tokens from sliding 14-day windows
    for start in range(0, 365-13, 7):
        x14 = lookback[:, start:start+14, :]
        _, _, wk0 = model.daily_encoder(x14)   # take most recent 7-day token
        week_tokens.append(wk0)

    week_fifo = torch.stack(week_tokens[-7:], dim=1)      # (B,7,H)

    # Generate processed weekly tokens → month FIFO
    month_tokens: List[torch.Tensor] = []
    wk_fifo = week_fifo.clone()
    for _ in range(12):
        _, wk_tok = model.weekly_encoder(wk_fifo)
        month_tokens.append(wk_tok)
        wk_fifo = torch.cat([wk_fifo[:,1:], wk_tok.unsqueeze(1)], 1)

    month_fifo = torch.stack(month_tokens[-12:], dim=1)   # (B,12,H)
    model.train()
    return week_fifo, month_fifo

def train():
    # 1. data
    df = merge_calendars("processed_data/calendar_scaled.parquet")
    train_loader, _, _ = build_loaders(df, batch_size=BATCH)  # only train loader
    NUM_F = len([c for c in df.columns if c not in ("date")])

    # 2. model, loss, optimiser
    model = HierForecastNet(NUM_F, HID).to(DEVICE)
    criterion = HierWRMSSE().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # 3. training loop
    for epoch in range(1, EPOCHS+1):
        model.train(); running = 0.0
        for lookback, x14, y_d, y_w, y_m in tqdm(train_loader, desc=f"Epoch {epoch}"):
            lookback = lookback.to(DEVICE); x14 = x14.to(DEVICE)
            y_d, y_w, y_m = [t.to(DEVICE) for t in (y_d, y_w, y_m)]

            # bootstrap context
            wk_fifo, mth_fifo = bootstrap_fifos(model, lookback)

            # forward & loss
            d_hat, w_hat, m_hat, *_ = model(x14, wk_fifo, mth_fifo)

            ins_d  = lookback[:,-15:-1,-1]
            ins_w  = lookback[:,-15:-1:7,-1]
            ins_m  = lookback[:,-365:,-1][:,None,:]

            loss = criterion(d_hat, w_hat, m_hat, y_d, y_w, y_m,
                             ins_d, ins_w, ins_m)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        avg = running/len(train_loader)
        print(f"Epoch {epoch:>3}  train WRMSSE {avg:.4f}")

        # checkpoint
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch:03}.pth")

if __name__ == "__main__":
    train()