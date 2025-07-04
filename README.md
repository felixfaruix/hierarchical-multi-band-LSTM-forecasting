# Hierarchical Multi-Band LSTM with Cross Attention for Ethanol Price Forecasting

This repository contains the implementation of a Hierarchical Attention Network (HAN) designed for forecasting monthly Ethanol T2 prices in Europe. The model utilizes a hierarchical multi-band LSTM architecture, enhanced by dual attention mechanisms (feature-level and temporal-level), to dynamically learn cross-variable interactions and temporal dependencies at daily, weekly, and monthly resolutions.

## Dataset and Features

The dataset covers daily observations starting from January 2010, including (with sources): 

* Ethanol volume (demand proxy): _https://www.barchart.com/futures/quotes/D2N25_
* Corn prices (feedstock cost): _https://www.barchart.com/futures/quotes/ZCN25_
* Brent crude oil prices (energy benchmark): _https://www.barchart.com/stocks/quotes/WTI_
* Foreign exchange rates (FX): _https://www.investing.com/currencies/usd-brl-historical-data_
* Producer price index (US) (PPI): _https://fred.stlouisfed.org/series/WPU06140341_
* Market closed indicator (binary flag): self implemented 

Additionally, engineered calendar features are included to capture cyclical temporal patterns:

* Day-of-week and month-of-year (sin/cos encoding)
* Week-of-month indicator
* End-of-month binary flag (`is_eom`)

## Model Architecture

The model consists of stacked recurrent LSTM blocks operating at three hierarchical time scales:

1. **Daily Level**: Input data passes through feature-level attention (highlighting important input variables each day), followed by a daily LSTM layer producing daily hidden states.
2. **Weekly Level**: Daily hidden states are compressed via temporal attention into weekly embeddings, which feed into a weekly LSTM.
3. **Monthly Level**: Weekly hidden states undergo a second round of temporal attention, forming monthly embeddings for the final monthly LSTM.

   flowchart TD
    subgraph INPUT["Multivariate Input"]
        X[ethanol, ethanol_volume, corn,<br> brent, fx, ppi,<br>market_closed,<br>calendar features]
    end

    subgraph DAILY["Daily Block"]
        FA[Feature-level Attention]<br>
        DLSTM[Daily LSTM - 14-day sequences]
    end

    subgraph WEEKLY["Weekly Block"]
        DA[Temporal Attention (Daily-to-Weekly)]<br>
        WLSTM[Weekly LSTM]
    end

    subgraph MONTHLY["Monthly Block"]
        WA[Temporal Attention (Weekly-to-Monthly)]<br>
        MLSTM[Monthly LSTM]<br>
        HEAD[Prediction Head]
    end

    X --> FA --> DLSTM --> DA --> WLSTM --> WA --> MLSTM --> HEAD

    style INPUT fill:#0d1117,stroke:#58a6ff,stroke-width:1px,color:#c9d1d9
    style DAILY fill:#161b22,stroke:#3fb950,stroke-width:1px,color:#c9d1d9
    style WEEKLY fill:#0d1117,stroke:#58a6ff,stroke-width:1px,color:#c9d1d9
    style MONTHLY fill:#161b22,stroke:#3fb950,stroke-width:1px,color:#c9d1d9

## Research Foundations and Inspiration

The architecture integrates key ideas from three influential papers:

* https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf: Inspired the hierarchical structure and the use of temporal attention for summarizing sequences across different time scales.

* https://link.springer.com/article/10.1007/s10614-025-11030-y?utm: Guided the integration of feature-level attention, emphasizing dynamic interactions between input variables.

* https://www.researchgate.net/publication/389106299_A_multi-energy_loads_forecasting_model_based_on_dual_attention_mechanism_and_multi-scale_hierarchical_residual_network_with_gated_recurrent_unit: It inspired the double cross-scale attention mechanism, enabling the model to focus effectively both on relevant features within each timestep and critical temporal points across different hierarchical scales.

Additionally, the model incorporates hierarchical reconciliation strategies as discussed by Rangapuram et al. (2023), ensuring coherent and consistent forecasts across multiple temporal resolutions through backpropagation of loss from the monthly prediction down to daily inputs.

## Prediction and Reconciliation

Final forecasts are generated at the monthly level, with loss propagation ensuring consistency across all time scales. Intermediate daily and weekly representations, though not directly used for prediction, significantly enhance the final monthly forecast quality through learned hierarchical embeddings.

---

**References:**

* Rangapuram, S. S., et al. (2023). "Cross-Scale Attention for Long-Term Time Series Forecasting." *Proceedings of the 40th International Conference on Machine Learning (ICML)*. [Paper Link](https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf)
* Zhou, Y., et al. (2025). "TimeCNN: Refining Cross-Variable Interaction on Time Point for Time Series Forecasting." *Computational Economics*. [Paper Link](https://link.springer.com/article/10.1007/s10614-025-11030-y?utm_source=chatgpt.com)
* A multi-energy loads forecasting model based on dual attention mechanism and multi-scale hierarchical residual network with gated recurrent unit. (2024). [Paper Link](https://www.researchgate.net/publication/389106299_A_multi-energy_loads_forecasting_model_based_on_dual_attention_mechanism_and_multi-scale_hierarchical_residual_network_with_gated_recurrent_unit)
