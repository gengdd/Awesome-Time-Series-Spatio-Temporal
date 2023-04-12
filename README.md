# Awesome-Time-Series-Spatio-Temporal

- [Time Series](#time-series)
  - [Time Series Forecasting](#time-series-forecasting)
  - [Time Series Anomaly Detection](#time-series-anomaly-detection)
  - [Time Series Representation Learning](#time-series-representation-learning)
  - [Time Series Imputation](#time-series-imputation)
- [Spatio-Temporal](#spatio-temporal)
  - [Spatio-Temporal Forecasting](#spatio-temporal-forecasting)
  - [Spatio-Temporal Imputation](#spatio-temporal-imputation)
- [Trajectory Data](#trajectory-data)
  - [Travel Time Estimation](#travel-time-estimation)
  - [Trajectory Prediction](#trajectory-prediction)
  - [Trajectory Representaion Learning](#trajectory-representaion-learning)
  - [Trajectory Anomaly Detection](#trajectory-anomaly-detection)
  - [Trajectory Recovery](#trajectory-recovery)

## Time Series

### Time Series Forecasting

#### ArXiv & Recently Updated

#### ICLR/ICML/NIPS

- [ICLR 2023] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.[Paper](https://openreview.net/pdf?id=Jbdc0vTOcol).[Code](https://github.com/yuqinie98/PatchTST)
- [ICLR 2023] TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.[Paper](https://openreview.net/pdf?id=ju_Uqw384Oq).[Code](https://github.com/thuml/Time-Series-Library)
- [ICLR 2023] MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting.[Paper](https://openreview.net/pdf?id=zt53IDUR1U).[Code](https://github.com/wanghq21/MICN)
- [ICLR 2023] Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.[Paper](https://openreview.net/pdf?id=vSVLM2j9eie).[Code](https://github.com/Thinklab-SJTU/Crossformer)
- [ICLR 2023] Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting. [Paper](https://openreview.net/pdf?id=sCrnllCtjoE). [Code](https://github.com/BorealisAI/scaleformer)
- [NIPS 2022] FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting. [Paper](https://arxiv.org/abs/2205.08897). [Code](https://github.com/tianzhou2011/FiLM/)
- [NIPS 2022] Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting. [Paper](https://arxiv.org/abs/2205.14415). [Code](https://github.com/thuml/Nonstationary_Transformers)
- [NIPS 2022] SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction. [Paper](https://arxiv.org/pdf/2106.09305.pdf). [Code](https://github.com/cure-lab/SCINet)
- [NIPS 2022] Generative Time Series Forecasting with **Diffusion**, Denoise, and Disentanglement. [Paper](https://arxiv.org/abs/2301.03028). [Code](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE)
- [ICML 2022] FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. [Paper](https://arxiv.org/abs/2201.12740). [Code](https://github.com/MAZiqing/FEDformer)
- [ICLR 2022] Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting.[Paper](https://openreview.net/pdf?id=0EXmFzUn5I).[Code](https://github.com/ant-research/Pyraformer)

#### AAAI/KDD/ICDE/VLDB/IJCAI

- [AAAI 2023] N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.[Paper](https://arxiv.org/abs/2201.12886).[Code](https://github.com/Nixtla/neuralforecast)
- [AAAI 2023] Are Transformers Effective for Time Series Forecasting?[Paper](https://arxiv.org/abs/2205.13504).[Code](https://github.com/cure-lab/LTSF-Linear)
- [AAAI 2023] Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting.[Paper](https://arxiv.org/abs/2302.14829).[Code](https://github.com/weifantt/Dish-TS)
- [KDD 2022] Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting. [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539234). [Code]

#### SigSpatial/CIKM/SIGMOD/WSDM/Others

### Time Series Anomaly Detection

#### ICLR/ICML/NIPS

- [ICLR 2023] Unsupervised Model Selection for Time Series Anomaly Detection. [Paper](https://openreview.net/pdf?id=gOZ_pKANaPW). [Code](https://github.com/mononitogoswami/tsad-model-selection)
- [ICML 2022] Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection. [Paper](https://proceedings.mlr.press/v162/chen22x/chen22x.pdf). [Code]
- [ICLR 2022] Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. [Paper](https://openreview.net/pdf?id=LzQQ89U1qm_). [Code](https://github.com/thuml/Anomaly-Transformer)
- [ICLR 2022] Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series. [Paper](https://arxiv.org/pdf/2202.07857.pdf). [Code](https://github.com/EnyanDai/GANF)

#### AAAI/KDD/ICDE/VLDB/IJCAI

- [KDD 2022] Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series. [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539117). [Code]

#### SigSpatial/CIKM/SIGMOD/WSDM/

### Time Series Representation Learning

#### ICLR/ICML/NIPS

- [ICLR 2023] Contrastive Learning for Unsupervised Domain Adaptation of Time Series.[Paper](https://openreview.net/pdf?id=xPkJYRsQGM).[Code](https://github.com/oezyurty/CLUDA)
- [NIPS 2022] Learning Latent Seasonal-Trend Representations for Time Series Forecasting. [Paper](https://openreview.net/forum?id=C9yUwd72yy). [Code](https://github.com/zhycs/LaST)
- [NIPS 2022] Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency. [Paper](https://arxiv.org/abs/2206.08496). [Code](https://github.com/mims-harvard/TFC-pretraining)
- [ICML 2022] Unsupervised Time-Series Representation Learning with Iterative Bilinear Temporal-Spectral Fusion. [Paper](https://arxiv.org/abs/2202.04770). [Code]
- [ICML 2022] Utilizing Expert Features for Contrastive Learning of Time-Series Representations. [Paper](https://arxiv.org/abs/2206.11517). [Code]
- [ICLR 2022] CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting. [Paper](https://openreview.net/pdf?id=PilZY3omXV2). [Code](https://github.com/salesforce/CoST)

#### AAAI/KDD/ICDE/VLDB/IJCAI

#### SigSpatial/CIKM/SIGMOD/WSDM/

### Time Series Imputation

#### ICLR/ICML/NIPS

- [ICLR 2023] Multivariate Time-series Imputation with Disentangled Temporal Representations.[Paper](https://openreview.net/pdf?id=rdjeCNUS6TG).[Code](https://github.com/liuwj2000/TIDER)
- [ICLR 2022] Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks. [Paper](https://arxiv.org/pdf/2108.00298.pdf). [Code](https://github.com/Graph-Machine-Learning-Group/grin)

#### AAAI/KDD/ICDE/VLDB/IJCAI

#### SigSpatial/CIKM/SIGMOD/WSDM/

## Spatio-Temporal

### Spatio-Temporal Forecasting

- [NIPS 2022] Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks. [Paper](https://papers.nips.cc/paper_files/paper/2022/hash/7b102c908e9404dd040599c65db4ce3e-Abstract-Conference.html). [Code](https://github.com/zyplanet/TPGNN)

### Spatio-Temporal Imputation

- [NIPS 2022] Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations. [Paper](https://arxiv.org/abs/2205.13479). [Code](https://github.com/Graph-Machine-Learning-Group/spin)
- 

## Trajectory Data

### Travel Time Estimation

### Trajectory Prediction

### Trajectory Representaion Learning

### Trajectory Anomaly Detection

### Trajectory Recovery
