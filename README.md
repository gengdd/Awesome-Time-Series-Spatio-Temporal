# Awesome-Time-Series-Spatio-Temporal

- [Time Series](#computer-time-series)
  - [Time Series Forecasting](#computer-time-series-forecasting)
  - [Time Series Anomaly Detection](#computer-time-series-anomaly-detection)
  - [Time Series Representation Learning](#computer-time-series-representation-learning)
  - [Time Series Imputation](#computer-time-series-imputation)
  - [Time Series Classification/Other](#computer-time-series-classificationother)
- [Spatio-Temporal](#computer-spatio-temporal)
  - [Spatio-Temporal Forecasting](#computer-spatio-temporal-forecasting)
  - [Spatio-Temporal Imputation](#computer-spatio-temporal-imputation)
- [Trajectory Data](#computer-trajectory-data)
  - [Travel Time Estimation](#computer-travel-time-estimation)
  - [Trajectory Prediction](#computer-trajectory-predictionpoi)
  - [Trajectory Representaion Learning](#computer-trajectory-representaion-learning)
  - [Trajectory Anomaly Detection](#computer-trajectory-anomaly-detection)
  - [Trajectory Generation/Recovery](#computer-trajectory-generationrecovery)
  - [Map Matching/Other](#computer-map-matchingother)
- [GeoAI/TransGPT](#computer-geoaitransgpt)

## :computer: Time Series

### :computer: Time Series Forecasting

- [CIKM 2023] GCformer: An Efficient Solution for Accurate and Scalable Long-Term Multivariate Time Series Forecasting. [Paper]. [Code]
- [CIKM 2023] MemDA: Forecasting Urban Time Series with Memory-based Drift Adaptation. [Paper]. [Code]
- [CIKM 2023] DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction. [Paper]. [Code]
- [CIKM 2023] FAMC-Net: Frequency Domain Parity Correction Attention and Multi-Scale Dilated Convolution for Time Series Forecasting. [Paper]. [Code]
- [VLDB 2023] SimpleTS: An Efficient and Universal Model Selection Framework for Time Series Forecasting. [Paper]. [Code]
- [KDD 2023] Sparse Binary Transformers for Multivariate Time Series Modeling. [Paper]. [Code]
- [KDD 2023] WHEN: A Wavelet-DTW Hybrid Attention Network for Heterogeneous Time Series Analysis. [Paper]. [Code]
- [KDD 2023] TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting. [Paper]. [Code]
- [KDD 2023] Hierarchical Proxy Modeling for Improved HPO in Time Series Forecasting. [Paper]. [Code]
- [ICML 2023] Learning Perturbations to Explain Time Series Predictions. [Paper]. [Code]
- [ICML 2023] Self-Interpretable Time Series Prediction with Counterfactual Explanations. [Paper]. [Code]
- [ICML 2023] Learning Deep Time-index Models for Time Series Forecasting. [Paper]. [Code]
- [ICML 2023] Feature Programming for Multivariate Time Series Prediction. [Paper]. [Code]
- [ICML 2023] Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting. [Paper]. [Code]
- [ICML 2023] Non-autoregressive Conditional Diffusion Models for Time Series Prediction. [Paper]. [Code]
- [IJCAL 2023] SMARTformer: Semi-Autoregressive Transformer with Efficient Integrated Window Attention for Long Time Series Forecasting. [Paper]. [Code]
- [IJCAI 2023] Latent Processes Identification From Multi-View Time Series. [Paper]. [Code]
- [IJCAI 2023] DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long-term Effects in Time Series. [Paper]. [Code]
- [IJCAI 2023] Learning Gaussian Mixture Representations for Tensor Time Series Forecasting. [Paper]. [Code]
- [IJCAI 2023] pTSE: A Multi-model Ensemble Method for Probabilistic Time Series Forecasting. [Paper]. [Code]
- [IJCAI 2023] Not Only Pairwise Relationships: Fine-Grained Relational Modeling for Multivariate Time Series Forecasting. [Paper]. [Code]
- [IJCAI 2023] Transformers in Time Series: A Survey. [Paper]. [Code]
- [AAAI 2023] SLOTH: Structured Learning and Task-Based Optimization for Time Series Forecasting on Hierarchies. [Paper]. [Code]
- [AAAI 2023] Learning Decomposed Spatial Relations for Multi-Variate Time-Series Modeling. [Paper]. [Code]
- [AAAI 2023] InParformer: Evolutionary Decomposition Transformers with Interactive Parallel Attention for Long-Term Time Series Forecasting. [Paper]. [Code]
- [AAAI 2023] Learning Dynamic Temporal Relations with Continuous Graph for Multivariate Time Series Forecasting. [Paper]. [Code]
- [AAAI 2023] WaveForM: Graph Enhanced Wavelet Learning for Long Sequence Forecasting of Multivariate Time Series. [Paper]. [Code]
- [AAAI 2023] N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. [Paper](https://arxiv.org/abs/2201.12886). [Code](https://github.com/Nixtla/neuralforecast)
- [AAAI 2023] Are Transformers Effective for Time Series Forecasting? [Paper](https://arxiv.org/abs/2205.13504). [Code](https://github.com/cure-lab/LTSF-Linear)
- [AAAI 2023] Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting. [Paper](https://arxiv.org/abs/2302.14829). [Code](https://github.com/weifantt/Dish-TS)
- [ICLR 2023] Learning Fast and Slow for Online Time Series Forecasting. [Paper]. [Code]
- [ICLR 2023] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [Paper](https://openreview.net/pdf?id=Jbdc0vTOcol). [Code](https://github.com/yuqinie98/PatchTST)
- [ICLR 2023] TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. [Paper](https://openreview.net/pdf?id=ju_Uqw384Oq). [Code](https://github.com/thuml/Time-Series-Library)
- [ICLR 2023] MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting. [Paper](https://openreview.net/pdf?id=zt53IDUR1U). [Code](https://github.com/wanghq21/MICN)
- [ICLR 2023] Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. [Paper](https://openreview.net/pdf?id=vSVLM2j9eie). [Code](https://github.com/Thinklab-SJTU/Crossformer)
- [ICLR 2023] Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting. [Paper](https://openreview.net/pdf?id=sCrnllCtjoE). [Code](https://github.com/BorealisAI/scaleformer)
- [ICDE 2023] Mining Seasonal Temporal Patterns in Time Series. [Paper]. [Code]
- [ICDE 2023] Towards Long-Term Time-Series Forecasting: Feature, Pattern, and Distribution. [Paper]. [Code]

<br>

- [NIPS 2022] FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting. [Paper](https://arxiv.org/abs/2205.08897). [Code](https://github.com/tianzhou2011/FiLM/)
- [NIPS 2022] Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting. [Paper](https://arxiv.org/abs/2205.14415). [Code](https://github.com/thuml/Nonstationary_Transformers)
- [NIPS 2022] SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction. [Paper](https://arxiv.org/pdf/2106.09305.pdf). [Code](https://github.com/cure-lab/SCINet)
- [NIPS 2022] Generative Time Series Forecasting with **Diffusion**, Denoise, and Disentanglement. [Paper](https://arxiv.org/abs/2301.03028). [Code](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE)
- [ICML 2022] FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. [Paper](https://arxiv.org/abs/2201.12740). [Code](https://github.com/MAZiqing/FEDformer)
- [ICLR 2022] Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting. [Paper](https://openreview.net/pdf?id=0EXmFzUn5I). [Code](https://github.com/ant-research/Pyraformer)
- [KDD 2022] Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting. [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539234). [Code]
- [KDD 2022] Learning Differential Operators for Interpretable Time Series Modeling. [Paper]. [Code]
- [CIKM 2022] Deep Extreme Mixture Model for Time Series Forecasting. [Paper]. [Code]

### :computer: Time Series Anomaly Detection

- [CIKM 2023] DuoGAT: Dual Time-oriented Graph Attention Networks for Accurate, Efficient and Explainable Anomaly Detection on Time-series. [Paper]. [Code]
- [VLDB 2023] Choose Wisely: An Extensive Evaluation of Model Selection for Anomaly Detection in Time Series. [Paper]. [Code]
- [VLDB 2023] OneShotSTL: One-Shot Seasonal-Trend Decomposition For Online Time Series Anomaly Detection And Forecasting. [Paper]. [Code]
- [KDD 2023] DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection. [Paper]. [Code]
- [KDD 2023] Precursor-of-Anomaly Detection for Irregular Time Series. [Paper]. [Code]
- [ICML 2023] Prototype-oriented unsupervised anomaly detection for multivariate time series. [Paper]. [Code]
- [AAAI 2023] Detecting Multivariate Time Series Anomalies with Zero Known Label. [Paper]. [Code]
- [ICLR 2023] Unsupervised Model Selection for Time Series Anomaly Detection. [Paper](https://openreview.net/pdf?id=gOZ_pKANaPW). [Code](https://github.com/mononitogoswami/tsad-model-selection)
- [WSDM 2023] Adversarial Autoencoder for Unsupervised Time Series Anomaly Detection and Interpretation. [Paper]. [Code]

<br>

- [ICML 2022] Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection. [Paper](https://proceedings.mlr.press/v162/chen22x/chen22x.pdf). [Code]
- [ICLR 2022] Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. [Paper](https://openreview.net/pdf?id=LzQQ89U1qm_). [Code](https://github.com/thuml/Anomaly-Transformer)
- [ICLR 2022] Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series. [Paper](https://arxiv.org/pdf/2202.07857.pdf). [Code](https://github.com/EnyanDai/GANF)
- [KDD 2022] Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series. [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539117). [Code]
- [CIKM 2022] TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Freq Analysis. [Paper]. [Code]

### :computer: Time Series Representation Learning

- [CIKM 2023] TriD-MAE: A Generic Pre-Train Model for Multivariate Time Series with Missing Values. [Paper]. [Code]
- [CIKM 2023] A Co-training Approach for Noisy Time Series Learning. [Paper]. [Code]
- [AAAI 2023] Time Series Contrastive Learning with Information-Aware Augmentations. [Paper]. [Code]
- [AAAI 2023] Supervised Contrastive Few-shot Learning for High-frequency Time Series. [Paper]. [Code]
- [AAAI 2023] Temporal-Frequency Co-Training for Time Series Semi-Supervised Learning. [Paper]. [Code]
- [ICLR 2023] Contrastive Learning for Unsupervised Domain Adaptation of Time Series. [Paper](https://openreview.net/pdf?id=xPkJYRsQGM). [Code](https://github.com/oezyurty/CLUDA)

<br>

- [NIPS 2022] Learning Latent Seasonal-Trend Representations for Time Series Forecasting. [Paper](https://openreview.net/forum?id=C9yUwd72yy). [Code](https://github.com/zhycs/LaST)
- [NIPS 2022] Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency. [Paper](https://arxiv.org/abs/2206.08496). [Code](https://github.com/mims-harvard/TFC-pretraining)
- [ICML 2022] Unsupervised Time-Series Representation Learning with Iterative Bilinear Temporal-Spectral Fusion. [Paper](https://arxiv.org/abs/2202.04770). [Code]
- [ICML 2022] Utilizing Expert Features for Contrastive Learning of Time-Series Representations. [Paper](https://arxiv.org/abs/2206.11517). [Code]
- [ICLR 2022] CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting. [Paper](https://openreview.net/pdf?id=PilZY3omXV2). [Code](https://github.com/salesforce/CoST)

### :computer: Time Series Imputation

- [CIKM 2023] Density-Aware Temporal Attentive Step-wise Diffusion Model For Medical Time Series Imputation. [Paper]. [Code]
- [KDD 2023] Source-Free Domain Adaptation with Temporal Imputation for Time Series Data. [Paper]. [Code]
- [KDD 2023] Networked Time Series Imputation via Position-aware Graph Enhanced Variational Autoencoders. [Paper]. [Code]
- [KDD 2023] Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models. [Paper]. [Code]
- [KDD 2023] An Observed Value Consistent Diffusion Model for Imputing Missing Values in Multivariate Time Series. [Paper]. [Code]
- [ICML 2023] Provably Convergent Schrödinger Bridge with Applications to Probabilistic Time Series Imputation. [Paper]. [Code]
- [ICML 2023] Deep Latent State Space Models for Time-Series Generation. [Paper]. [Code]
- [ICLR 2023] Multivariate Time-series Imputation with Disentangled Temporal Representations. [Paper](https://openreview.net/pdf?id=rdjeCNUS6TG). [Code](https://github.com/liuwj2000/TIDER)
  
<br>

- [ICLR 2022] Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks. [Paper](https://arxiv.org/pdf/2108.00298.pdf). [Code](https://github.com/Graph-Machine-Learning-Group/grin)

### :computer: Time Series Classification/Other

- [CIKM 2023] Time-series Shapelets with Learnable Lengths. [Paper]. [Code]
- [CIKM 2023] Temporal Convolutional Explorer Helps Understand 1D-CNN's Learning Behavior in Time Series Classification from Frequency Domain. [Paper]. [Code]
- [CIKM 2023] EFFECTS: Explorable and Explainable Feature Extraction Framework for Multivariate Time-Series Classification. [Paper]. [Code]
- [VLDB 2023] Motiflets - Simple and Accurate Detection of Motifs in Time Series. [Paper]. [Code]
- [VLDB 2023] Fast and Scalable Mining of Time Series Motifs with Probabilistic Guarantees. [Paper]. [Code]
- [VLDB 2023] Time2Feat: Learning Interpretable Representations for Multivariate Time Series Clustering. [Paper]. [Code]
- [KDD 2023] Self-supervised Classification of Clinical Multivariate Time Series using Time Series Dynamics. [Paper]. [Code]
- [KDD 2023] FLAMES2Graph: An Interpretable Federated Multivariate Time Series Classification Framework. [Paper]. [Code]
- [WWW 2023] FormerTime: Hierarchical Multi-Scale Representations for Multivariate Time Series Classification. [Paper]. [Code]
- [ICLR 2023] Out-of-distribution Representation Learning for Time Series Classification. [Paper]. [Code]

## :computer: Spatio-Temporal

### :computer: Spatio-Temporal Forecasting

- [Arvix] Spatio-Temporal Graph Neural Networks for Predictive Learning in Urban Computing: A Survey. [Paper]. [Code]

<br>

- [CIKM 2023] STREAMS: Towards Spatio-Temporal Causal Discovery with Reinforcement Learning for Streamflow Rate Prediction. [Paper]. [Code]
- [CIKM 2023] Prompt-Enhanced Spatio-Temporal Multi-Attribute Prediction. [Paper]. [Code]
- [CIKM 2023] GraphERT-- Transformers-based Temporal Dynamic Graph Embedding. [Paper]. [Code]
- [CIKM 2023] Region Profile Enhanced Urban Spatio-Temporal Prediction via Adaptive Meta Learning. [Paper]. [Code]
- [CIKM 2023] Mask- and Contrast-Enhanced Spatio-Temporal Learning for Urban Flow Prediction. [Paper]. [Code]
- [CIKM 2023] Enhancing Spatio-temporal Traffic Prediction through Urban Human Activity Analysis. [Paper]. [Code]
- [CIKM 2023] ST-MoE: Spatio-Temporal Mixture-of-Experts for Debiasing in Traffic Prediction. [Paper]. [Code]
- [CIKM 2023] Enhancing the Robustness via Adversarial Learning and Joint Spatial-temporal Embeddings in Traffic Forecasting. [Paper]. [Code]
- [CIKM 2023] MLPST: MLP is All You Need for Spatio-Temporal Prediction. [Paper]. [Code]
- [CIKM 2023] Spatial-temporal Graph Boosting Network: Enhancing Spatial-temporal Graph Neural Networks via Gradient Boosting. [Paper]. [Code]
- [CIKM 2023] Spatial-Temporal-Aware Meta Graph Contrastive Learning. [Paper]. [Code]
- [CIKM 2023] Explainable Spatial-Temporal Graph Neural Networks. [Paper]. [Code]
- [CIKM 2023] Cross-city Few-Shot Traffic Forecasting via Traffic Pattern Bank. [Paper]. [Code]
- [CIKM 2023] Hierarchical Information Enhanced Traffic Forecasting. [Paper]. [Code]
- [CIKM 2023] CARPG: Cross-City Knowledge Transfer for Traffic Accident Prediction via Attentive Region-level Parameter Generation. [Paper]. [Code]
- [CIKM 2023] Time-aware Graph Structure Learning via Sequence Prediction on Temporal Graphs. [Paper]. [Code]
- [CIKM 2023] DeepSTA: A Spatial-Temporal Attention Network for Logistics Delivery Timely Rate Prediction in Anomaly Conditions. [Paper]. [Code]
- [KDD 2023] Graph Neural Processes for Spatio-Temporal Extrapolation. [Paper]. [Code]
- [KDD 2023] Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities. [Paper]. [Code]
- [KDD 2023] Robust Spatiotemporal Traffic Forecasting with Reinforced Dynamic Adversarial Training. [Paper]. [Code]
- [KDD 2023] Pattern Expansion and Consolidation on Evolving Graphs for Continual Traffic Prediction. [Paper]. [Code]
- [KDD 2023] TransformerLight: A Novel Sequence Modeling Based Traffic Signaling Mechanism via Gated Transformer. [Paper]. [Code]
- [KDD 2023] Spatial Heterophily Aware Graph Neural Networks. [Paper]. [Code]
- [KDD 2023] Multi-Temporal Relationship Inference in Urban Areas. [Paper]. [Code]
- [KDD 2023] Frigate: Frugal Spatio-temporal Forecasting on Road Networks. [Paper]. [Code]
- [KDD 2023] Maintaining the Status Quo: Capturing Invariant Relations for OOD Spatiotemporal Learning. [Paper]. [Code]
- [KDD 2023] Localised Adaptive Spatial-Temporal Graph Neural Network. [Paper]. [Code]
- [KDD 2023] A Data-driven Region Generation Framework for Spatiotemporal Transportation Service Management. [Paper]. [Code]
- [KDD 2023] Large-scale Urban Cellular Traffic Generation via Knowledge-Enhanced GANs with Multi-Periodic Patterns. [Paper]. [Code]
- [KDD 2023] Deep Transfer Learning for City-scale Cellular Traffic Generation through Urban Knowledge Graph. [Paper]. [Code]
- [AAAI 2023] Ising-Traffic: Using Ising Machine Learning to Predict Traffic Congestion under Uncertainty. [Paper]. [Code]
- [AAAI 2023] PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction. [Paper]. [Code]
- [AAAI 2023] Spatio-Temporal Graph Neural Point Process for Traffic Congestion Event Prediction. [Paper]. [Code]
- [AAAI 2023] Spatio-temporal Neural Structural Causal Models for Bike Flow Prediction. [Paper]. [Code]
- [AAAI 2023] Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction. [Paper]. [Code]
- [AAAI 2023] Spatio-Temporal Meta-Graph Learning for Traffic Forecasting. [Paper]. [Code]
- [AAAI 2023] Trafformer: Unify Time and Space in Traffic Prediction. [Paper]. [Code]
- [ICDE 2023] When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks. [Paper]. [Code]
- [ICDE 2023] Self-Supervised Spatial-Temporal Bottleneck Attentive Network for Efficient Long-term Traffic Forecasting. [Paper]. [Code]
- [ICDE 2023] Dynamic Hypergraph Structure Learning for Traffic Flow Forecasting. [Paper]. [Code]
- [WSDM 2023] A Multi-graph Fusion Based Spatiotemporal Dynamic Learning Framework. [Paper]. [Code]

<br>

- [NIPS 2022] Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks. [Paper](https://papers.nips.cc/paper_files/paper/2022/hash/7b102c908e9404dd040599c65db4ce3e-Abstract-Conference.html). [Code](https://github.com/zyplanet/TPGNN)
- [NIPS 2022] Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models. [Paper]. [Code]
- [ICML 2022] DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting. [Paper]. [Code]
- [KDD 2022] MSDR: Multi-Step Dependency Relation Networks for Spatial Temporal Forecasting. [Paper]. [Code]
- [KDD 2022] Modeling Network-level Traffic Flow Transitions on Sparse Data. [Paper]. [Code]
- [CIKM 2022] Automated Spatio-Temporal Synchronous Modeling with Multiple Graphs for Traffic Prediction. [Paper]. [Code]
- [CIKM 2022] Domain Adversarial Spatial-Temporal Network: A Transferable Framework for Short-term Traffic Forecasting across Cities. [Paper]. [Code]
- [Sigspatial 2022] When Do Contrastive Learning Signals Help Spatio-Temporal Graph Forecasting? [Paper]. [Code]

### :computer: Spatio-Temporal Imputation

- [ICDE 2023] PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. [Paper]. [Code]

<br>

- [NIPS 2022] Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations. [Paper](https://arxiv.org/abs/2205.13479). [Code](https://github.com/Graph-Machine-Learning-Group/spin)
- [CIKM 2022] Traffic Speed Imputation with Spatio-Temporal Attentions and Cycle-Perceptual Training. [Paper]. [Code]
- [SigSpatial 2022] Spatio-Temporal Graph Convolutional Network for Stochastic Traffic Speed Imputation. [Paper]. [Code]

## :computer: Trajectory Data

### :computer: Travel Time Estimation

- [CIKM 2023] GBTTE: Graph Attention Network Based Bus Travel Time Estimation. [Paper]. [Code]
- [CIKM 2023] HST-GT:Heterogeneous Spatial-Temporal Graph Transformer for Delivery Time Estimation in Warehouse-Distribution Integration E-Commerce. [Paper]. [Code]
- [CIKM 2023] Fragment and Integrate Network (FIN): A Novel Spatial-Temporal Modeling Based on Long Sequential Behavior for Online Food Ordering Click-Through Rate Prediction. [Paper]. [Code]
- [VLDB 2023] Route Travel Time Estimation on A Road Network Revisited: Heterogeneity, Proximity, Periodicity and Dynamicity. [Paper]. [Code]
- [KDD 2023] Uncertainty-Aware Probabilistic Travel Time Prediction for On-Demand Ride-Hailing at DiDi. [Paper]. [Code]
- [KDD 2023] iETA: A Robust and Scalable Incremental Learning Framework for Time-of-Arrival Estimation. [Paper]. [Code]
- [ICDE 2023] Delivery Time Prediction Using Large-Scale Graph Structure Learning Based on Quantile Regression. [Paper]. [Code]
- [WSDM 2023] Inductive Graph Transformer for Delivery Time Estimation. [Paper]. [Code]

<br>

- [KDD 2022] Reproducibility and Progress in Estimating Time of Arrival, or Can Simple Methods Still Outperform Deep Learning Ones? [Paper]. [Code]
- [KDD 2022] Interpreting Trajectories from Multiple Views: A Hierarchical Self-Attention Network for Estimating the Time of Arrival. [Paper]. [Code]
- [SigSpatial 2022] MTTPRE: A Multi-Scale Spatial-Temporal Model for Travel Time Prediction. [Paper]. [Code]

### :computer: Trajectory Prediction/POI

- [SIGIR 2023] Spatio-Temporal Hypergraph Learning for Next POI Recommendation. [Paper]. [Code]
- [SIGIR 2023] Adaptive Graph Representation Learning for Next POI Recommendation. [Paper]. [Code]
- [SIGIR 2023] EEDN: Enhanced Encoder-Decoder Network with Local and Global Context Learning for POI Recommendation. [Paper]. [Code]
- [AAAI 2023] Causal Intervention for Human Trajectory Prediction with Cross Attention Mechanism. [Paper]. [Code]
- [AAAI 2023] Mobility Prediction via Sequential Trajectory Disentanglement. [Paper]. [Code]
- [AAAI 2023] WSiP: Wave Superposition Inspired Pooling for Dynamic InteractionsAware Trajectory Prediction. [Paper]. [Code]

<br>

- [KDD 2022] MetaPTP: An Adaptive Meta-optimized Model for Personalized Spatial Trajectory Prediction. [Paper]. [Code]
- [KDD 2022] Graph2Route: A Dynamic Spatial-Temporal Graph Neural Network for Pick-up and Delivery Route Prediction. [Paper]. [Code]

### :computer: Trajectory Representaion Learning

- [VLDB 2023] A Deep Generative Model for Trajectory Modeling and Utilization. [Paper]. [Code]
- [KDD 2023] LightPath: Lightweight and Scalable Path Representation Learning. [Paper]. [Code]
- [ICDE 2023] Self-supervised Trajectory Representation Learning with Temporal Regularities and Travel Semantics. [Paper]. [Code]
- [ICDE 2023] BERT-Trip: Effective and Scalable Trip Representation using Attentive Contrast Learning. [Paper]. [Code]

<br>

- [CIKM 2022] Jointly Contrastive Representation Learning on Road Network and Trajectory. [Paper]. [Code]

### :computer: Trajectory Similarity Computation

- [AAAI 2023] GRLSTM: Trajectory Similarity Computation with Graph-based Residual LSTM. [Paper]. [Code]
- [ICDE 2023] Contrastive Trajectory Similarity Learning with Dual-Feature Attention. [Paper]. [Code]

<br>

- [KDD 2022] A Graph-based Approach for Trajectory Similarity Computation in Spatial Networks. [Paper]. [Code]
- [KDD 2022] TrajGAT: A Graph-based Long-term Dependency Modeling Approach for Trajectory Similarity Computation. [Paper]. [Code]
- [KDD 2022] Spatio-Temporal Trajectory Similarity Learning in Road Networks. [Paper]. [Code]
- [CIKM 2022] Can Adversarial Training benefit Trajectory Representation? An Investigation on Robustness for Trajectory Similarity Computation. [Paper]. [Code]
- [CIKM 2022] Efficient Trajectory Similarity Computation with Contrastive Learning. [Paper]. [Code]

### :computer: Trajectory Anomaly Detection

- [ICDE 2023] Online Anomalous Subtrajectory Detection on Road Networks with Deep Reinforcement Learning. [Paper]. [Code]

### :computer: Trajectory Classification

- [CIKM 2022] TrajFormer: Efficient Trajectory Classification with Transformers.


### :computer: Trajectory Generation/Recovery

- [AAAI 2023] Continuous Trajectory Generation Based on Two-Stage GAN. [Paper]. [Code]
- [AAAI 2023] PateGail: A Privacy-preserving Mobility Trajectory Generator with Imitation Learning. [Paper]. [Code]
- [ICDE 2023] RNTrajRec: Road Network Enhanced Trajectory Recovery with Spatial-Temporal Transformer. [Paper]. [Code]

<br>

- [KDD 2022] Spatio-Temporal Vehicle Trajectory Recovery on Road Network Based on Traffic Camera Video Data. [Paper]. [Code]
- [SigSpatial 2022] Factorized Deep Generative Models for End-to-End Trajectory Generation with Spatiotemporal Validity Constraints. [Paper]. [Code]

### :computer: Map Matching/Other

- [VLDB 2023] LDPTrace: Locally Differentially Private Trajectory Synthesis. [Paper]. [Code]
- [VLDB 2023] Trajectory Data Collection with Local Differential Privacy. [Paper]. [Code]
- [VLDB 2023] Efficient Non-Learning Similar Subtrajectory Search. [Paper]. [Code]
- [KDD 2023] Understanding the Semantics of GPS-based Trajectories for Road Closure Detection. [Paper]. [Code]
- [ICDE 2023] LHMM: A Learning Enhanced HMM Model for Cellular Trajectory Map-matching. [Paper]. [Code]

## :computer: GeoAI/TransGPT

- [SIGIR 2023] MGeo: Multi-Modal Geographic Language Model Pre-Training. [Paper]. [Code]
