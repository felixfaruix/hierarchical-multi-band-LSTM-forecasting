"""
Hierarchical Forecasting Model using Attention and RNNs with PyTorch.
This module implements a hierarchical forecasting model that predicts daily, weekly, and monthly values from time series data.
It uses a combination of attention mechanisms and recurrent neural networks to process sequences of features and generate forecasts.    
It is designed to work with a DataFrame that has been preprocessed to include the necessary features and target variable.
    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        features (List[str]): List of feature column names to be used for forecasting.
    Returns:
        List[str]: List of feature columns excluding 'date'.
    
   return [col for col in df.columns if col != "date"]

It's divided into several components:
1. **FeatureAttention**: Soft-selects features within a single day using a linear layer
2. **ChunkAttentionPool**: Applies multi-head attention pooling over a 7-step chunk to produce a chunk token.
3. **DailyEncoder**: Encodes a 14-day sequence into weekly tokens and a next-day forecast.
4. **WeeklyEncoder**: Encodes weekly tokens to produce a next-week forecast and a weekly token.
5. **MonthlyEncoder**: Encodes monthly tokens to produce a month token.
6. **MonthDecoder**: Autoregressively decodes 30 daily values from the month token.
7. **HierForecastNet**: The main model that combines all encoders and decoders

Between the components, there are utility layers like `PreNormRes` for residual connections with layer normalization, 
and `GLUffn` for feed-forward networks with gated linear units.

"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor


class FeatureAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
    def forward(self, x: Tensor) -> Tensor:
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)

class ChunkAttentionPool(nn.Module):
    def __init__(self, hid: int, heads: int = 4, p: float = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, hid))
        self.mha = nn.MultiheadAttention(hid, heads, dropout=p, batch_first=True)
    def forward(self, chunk: Tensor) -> Tensor:
        q = self.q.expand(chunk.size(0), -1, -1)
        ctx, _ = self.mha(q, chunk, chunk)
        return ctx.squeeze(1)

class PreNormRes(nn.Module):
    def __init__(self, dim: int, sub: nn.Module, p: float = 0.1):
        super().__init__()
        self.norm, self.sub, self.drop = nn.LayerNorm(dim), sub, nn.Dropout(p)
    def forward(self, x: Tensor, *a, **k):
        return x + self.drop(self.sub(self.norm(x), *a, **k))

class GLUffn(nn.Module):
    def __init__(self, dim: int, p: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GLU(), nn.Dropout(p), nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.ff(x)

class DailyEncoder(nn.Module):
    def __init__(self, in_f: int, hid: int, p: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_f, hid)
        self.day_feat_attn = FeatureAttention(in_f)
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.chunk_pool = ChunkAttentionPool(hid)
        self.day_head = nn.Linear(hid, 1)
    def forward(self, x14: Tensor):
        B, _, F = x14.shape
        xw = x14.view(B * 14, F)
        xf = self.day_feat_attn(xw).view(B, 14, 1)
        x = self.proj(x14) * xf
        seq, _ = self.lstm(x)
        day_pred = self.day_head(seq[:, -1])
        wk1_tok = self.chunk_pool(seq[:, 0:7])
        wk0_tok = self.chunk_pool(seq[:, 7:14])
        return day_pred.squeeze(1), wk1_tok, wk0_tok

class WeeklyEncoder(nn.Module):
    """7 weekly tokens → **7‑day vector** + new weekly token."""
    def __init__(self, hid: int, heads: int = 4, p: float = 0.1):
        super().__init__()
        self.cross_attn = PreNormRes(
            hid, nn.MultiheadAttention(hid, heads, dropout=p, batch_first=True), p)
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.ffn = PreNormRes(hid, GLUffn(hid, p), p)
        self.week_head = nn.Linear(hid, 7)  # <-- return 7‑vector
    def forward(self, wtokens: Tensor):
        q = wtokens[:, -1:, :]
        ctx, _ = self.cross_attn.sub(q, wtokens, wtokens)
        seq, _ = self.lstm(torch.cat([wtokens, ctx], dim=1))
        token = self.ffn(seq[:, -1])
        wk_pred = self.week_head(token)  # [B,7]
        return wk_pred, token

class MonthlyEncoder(nn.Module):
    def __init__(self, hid: int, heads: int = 4, p: float = 0.1):
        super().__init__()
        self.cross_attn = PreNormRes(
            hid, nn.MultiheadAttention(hid, heads, dropout=p, batch_first=True), p)
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.ffn = PreNormRes(hid, GLUffn(hid, p), p)
    def forward(self, mtokens: Tensor):
        q = mtokens[:, -1:, :]
        ctx, _ = self.cross_attn.sub(q, mtokens, mtokens)
        seq, _ = self.lstm(torch.cat([mtokens, ctx], dim=1))
        return self.ffn(seq[:, -1])

class MonthDecoder(nn.Module):
    def __init__(self, hid: int, p: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.out = nn.Linear(hid, 1)
        self.drop = nn.Dropout(p)
    def forward(self, token: Tensor, steps: int = 30):
        B, H = token.shape
        h = token.unsqueeze(0); c = torch.zeros_like(h)
        inp = torch.zeros(B, 1, H, device=token.device)
        outs = []
        for _ in range(steps):
            out, (h, c) = self.lstm(inp, (h, c))
            val = self.out(self.drop(out))
            outs.append(val.squeeze(2))
            inp = out
        return torch.cat(outs, dim=1)

# ---------- full model ----------
class HierForecastNet(nn.Module):
    def __init__(self, in_f: int, hid: int = 128, p: float = 0.1):
        super().__init__()
        self.daily_enc = DailyEncoder(in_f, hid, p)
        self.week_enc = WeeklyEncoder(hid, p=p)
        self.month_enc = MonthlyEncoder(hid, p=p)
        self.month_dec = MonthDecoder(hid, p)
    def forward(self, x14: Tensor, week_fifo: Tensor, month_fifo: Tensor):
        day_pred, wk1_tok, wk0_tok = self.daily_enc(x14)
        week_input = torch.cat([week_fifo[:, 1:], wk0_tok.unsqueeze(1)], dim=1)
        week_pred, week_tok = self.week_enc(week_input)
        month_input = torch.cat([month_fifo[:, 1:], week_tok.unsqueeze(1)], dim=1)
        month_token = self.month_enc(month_input)
        month_pred_seq = self.month_dec(month_token)
        return day_pred, week_pred, month_pred_seq, wk0_tok, week_tok
