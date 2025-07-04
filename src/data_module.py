import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class WindowDataset(Dataset):

    def __init__(self, df: pd.DataFrame,  feature_cols: list, target_col: str = "ethanol", window_size: int = 14):
        self.window = window_size
        self.X = df[feature_cols].values.astype("float32")
        self.y = df[target_col].values.astype("float32")

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x_win = self.X[idx : idx + self.window]
        y_next = self.y[idx + self.window]
        return torch.from_numpy(x_win), torch.tensor(y_next)