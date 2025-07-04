# Hierarchical Multi-Band LSTM with Cross Attention for Ethanol Price Forecasting

This repository contains the implementation of a Hierarchical Attention Network (HAN) designed for forecasting monthly Ethanol T2 prices in Europe. The model utilizes a hierarchical multi-band LSTM architecture, enhanced by dual attention mechanisms (feature-level and temporal-level), to dynamically learn cross-variable interactions and temporal dependencies at daily, weekly, and monthly resolutions.

## Dataset and Features

The dataset covers daily observations starting from January 2010, including:

* Ethanol volume (demand proxy)
* Corn prices (feedstock cost)
* Brent crude oil prices (energy benchmark)
* Foreign exchange rates (FX)
* Producer price index (PPI)
* Market closed indicator (binary flag)

Additionally, engineered calendar features are included to capture cyclical temporal patterns:

* Day-of-week and month-of-year (sin/cos encoding)
* Week-of-month indicator
* End-of-month binary flag (`is_eom`)

## Model Architecture

The model consists of stacked recurrent LSTM blocks operating at three hierarchical time scales:

1. **Daily Level**: Input data passes through feature-level attention (highlighting important input variables each day), followed by a daily LSTM layer producing daily hidden states.
2. **Weekly Level**: Daily hidden states are compressed via temporal attention into weekly embeddings, which feed into a weekly LSTM.
3. **Monthly Level**: Weekly hidden states undergo a second round of temporal attention, forming monthly embeddings for the final monthly LSTM.

## Research Foundations and Inspiration

The architecture integrates key ideas from two influential papers:

* https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf: Inspired the hierarchical structure and the use of temporal attention for summarizing sequences across different time scales.

* https://link.springer.com/article/10.1007/s10614-025-11030-y?utm_source=chatgpt.com: Guided the integration of feature-level attention, emphasizing dynamic interactions between input variables.

Additionally, the model incorporates hierarchical reconciliation strategies as discussed by Rangapuram et al. (2023), ensuring coherent and consistent forecasts across multiple temporal resolutions through backpropagation of loss from the monthly prediction down to daily inputs.

## Prediction and Reconciliation

Final forecasts are generated at the monthly level, with loss propagation ensuring consistency across all time scales. Intermediate daily and weekly representations, though not directly used for prediction, significantly enhance the final monthly forecast quality through learned hierarchical embeddings.

---

**References:**

* Rangapuram, S. S., et al. (2023). "Cross-Scale Attention for Long-Term Time Series Forecasting." *Proceedings of the 40th International Conference on Machine Learning (ICML)*. [Paper Link](https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf)
* Zhou, Y., et al. (2025). "TimeCNN: Refining Cross-Variable Interaction on Time Point for Time Series Forecasting." *Computational Economics*. [Paper Link](https://link.springer.com/article/10.1007/s10614-025-11030-y?utm_source=chatgpt.com)
