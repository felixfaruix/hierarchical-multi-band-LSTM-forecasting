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

1. **Daily Level**: Input data passes through feature-level attention (emphasizing important input variables within each day), followed by a daily LSTM layer producing daily hidden states.
2. **Weekly Level**: Daily hidden states are aggregated using temporal attention and pooling into weekly embeddings, which feed into a weekly LSTM.
3. **Monthly Level**: Weekly hidden states undergo a second round of temporal attention, forming monthly embeddings for the final monthly LSTM.

### Attention, Pooling, and Lookback Memory (`timeseries_datamodule.py`)

The model uses a **sliding window approach**, enabling faster computation and  better handling of concept drifts by continuously updating the model with recent data (*Zliobaite et al., 2016*). It has a **1-year historical lookback tensor** (365 days). This memory allows attention layers to effectively utilize historical context, enriching the information passed to weekly and monthly blocks.

Inspired by *Rangapuram et al. (2023)*, the cross-scale attention mechanism enables to use shorter sliding windows compared to plain transformers, though not less than one full seasonal cycle (16 weeks for weekly, 12 months for monthly). In this way we enrich with much more seasonality pattern information. 

No traditional warm-up periods were implemented since the comprehensive year lookback already gives enough context for immediate monthly-level forecasting.

## Data Splitting and Handling

We split the dataset chronologically into training, validation, and test sets:

* **Training Set**: January 2010 to December 2021
* **Validation Set**: January 2022 to December 2022
* **Test Set**: January 2023 onwards

Training samples are shuffled to enhance gradient descent optimization, ensuring stable and robust learning. Validation and test sets are not shuffled to maintain chronological order, essential for accurate performance evaluation.

## Research Foundations and Inspiration

This architecture integrates ideas from:

* **Cross-Scale Transformer (Rangapuram et al., 2023)**: Influenced hierarchical attention layers and validated sliding window efficacy.
* **TimeCNN (Zhou et al., 2025)**: Guided dynamic cross-variable interaction modeling via feature-level attention.
* **Dual Attention Mechanism and Multi-scale Hierarchical Residual Networks (2024)**: Inspired our dual cross-scale attention mechanism, effectively prioritizing features and temporal points across scales.

The model also includes hierarchical reconciliation strategies (Rangapuram et al., 2023), ensuring consistent forecasts across all temporal resolutions via hierarchical loss propagation.

## Prediction and Reconciliation

Predictions are primarily monthly. Intermediate daily and weekly embeddings significantly enhance forecast accuracy through learned hierarchical representations. Reconciliation ensures internal consistency across all time scales.

---

**References:**

* Rangapuram, S. S., et al. (2023). "Cross-Scale Attention for Long-Term Time Series Forecasting." *ICML 2023*. [Paper](https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf)
* Zhou, Y., et al. (2025). "TimeCNN: Refining Cross-Variable Interaction on Time Point for Time Series Forecasting." *Computational Economics*. [Paper](https://link.springer.com/article/10.1007/s10614-025-11030-y?utm_source=chatgpt.com)
* "A multi-energy loads forecasting model based on dual attention mechanism and multi-scale hierarchical residual network with gated recurrent unit." (2024). [Paper](https://www.researchgate.net/publication/389106299_A_multi-energy_loads_forecasting_model_based_on_dual_attention_mechanism_and_multi-scale_hierarchical_residual_network_with_gated_recurrent_unit)
* Zliobaite, I., Pechenizkiy, M., & Gama, J. (2016). "An overview of concept drift applications." *Big Data Analysis: New Algorithms for a New Society*, 91-114.