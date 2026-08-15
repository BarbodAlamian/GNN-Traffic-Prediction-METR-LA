# GNN-based Traffic Speed Prediction on METR-LA Dataset

This repository contains the implementation of a **Graph Convolutional Network (GCN)** for spatio-temporal traffic speed forecasting using the **METR-LA** dataset. The primary objective is to evaluate the impact of spatial graph information on prediction accuracy and analyze various factors affecting model performance, such as graph depth, edge weighting, and sensor topology.

## Project Overview

Traffic speed prediction is a classic spatio-temporal problem. While temporal patterns (e.g., rush hour) are crucial, spatial dependencies (traffic flow propagating between neighboring roads) are equally important. This project implements a spatial GCN to capture these dependencies and compares it against non-graph baselines.

### Key Features
- **Graph Construction:** Builds both unweighted and distance-weighted adjacency graphs from 207 LA road sensors.
- **Model Architecture:** Implements a multi-layer Graph Convolutional Network (GCNConv) with ReLU activation and Dropout.
- **Comprehensive Analysis:** Conducts four distinct experiments to isolate the effects of spatial information, model depth, edge weights, and sensor locations.

## Dataset

The METR-LA dataset contains traffic speed readings collected from **207 loop detectors** on highways in Los Angeles County.

- **Time Span:** March – June 2012
- **Sampling Frequency:** 5-minute intervals
- **Total Timesteps:** 34,272
- **Graph Edges:** 1,515 (based on road network distances)
- **Data Split:** 70% Training, 10% Validation, 20% Test

## Experiments and Results

Four core experiments were designed to rigorously test the GCN's capabilities.

### 1. GCN vs. Non-Graph Models
We compared the GCN against a standard **MLP** (which uses flattened historical time-series data) and a **Historical Average** baseline.

| Model | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- |
| Historical Avg | 0.5392 | 0.8495 | Simple baseline |
| **MLP** | **0.2973** | **0.5303** | Uses 12 timesteps (1 hour) of history |
| GCN | 0.5264 | 0.8425 | Uses spatial graph but limited temporal input |

**Conclusion:** Although the GCN outperforms the Historical Average, it underperforms compared to the MLP. This is primarily because the MLP utilizes **all 12 historical timesteps**, whereas the current GCN implementation processes the input differently, highlighting that temporal features are currently more dominant than spatial features in this specific setup.

### 2. Effect of GCN Depth (Over-smoothing Analysis)
We tested GCN depths ranging from 1 to 4 layers to investigate the over-smoothing phenomenon.

| Layers | MAE | RMSE | Avg. Cosine Similarity |
| :--- | :--- | :--- | :--- |
| 1 | 0.5272 | 0.8425 | 0.1889 |
| **2** | **0.5270** | **0.8422** | 0.2868 |
| 3 | 0.5319 | 0.8537 | 0.3425 |
| 4 | 0.5334 | 0.8590 | 0.2949 |

**Conclusion:** A depth of **2 layers** provides the best balance. Increasing depth beyond 2 layers leads to performance degradation due to **Over-smoothing**, where node features become excessively mixed and lose their distinctiveness.

### 3. Weighted vs. Unweighted Graph
We evaluated whether distance-based edge weights (closer sensors = higher weight) improve performance.

| Graph Type | MAE | RMSE |
| :--- | :--- | :--- |
| Unweighted | 0.5264 | 0.8425 |
| **Weighted** | **0.4822** | **0.7812** |

**Conclusion:** The weighted graph significantly improves performance by **~8.39%** in MAE. This aligns with the physical intuition that geographically closer sensors exhibit more correlated traffic behavior.

### 4. Spatial Error Analysis
We analyzed prediction errors across individual sensors to identify spatial biases.

| Metric | Value |
| :--- | :--- |
| Mean Sensor Error | 0.5264 |
| Std. Deviation | 0.2597 |
| Max Error (Sensor 56) | 2.0966 |
| Min Error (Sensor 201) | 0.1773 |
| **Central Sensors Error** | **0.6174** |
| **Peripheral Sensors Error** | **0.4997** |

**Conclusion:** Central sensors (higher degree in the graph) exhibit significantly higher prediction errors compared to peripheral sensors. This is likely due to higher traffic congestion and noise aggregation in central nodes, making them inherently harder to predict.

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### Repository Structure

```text
.
├── A6_GNN_Barbod_Alamian.ipynb   # Main Jupyter Notebook
├── distances_la_2012.csv         # Distance matrix between sensors
├── graph_sensor_ids.txt          # List of sensor IDs
├── adj_mx.pkl                    # Precomputed adjacency matrix
├── metr-la.h5                    # Traffic speed dataset
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BarbodAlamian/GNN-Traffic-Prediction-METR-LA.git
   cd GNN-Traffic-Prediction-METR-LA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   Open `A6_GNN_Barbod_Alamian.ipynb` in Jupyter Lab, VS Code, or your preferred environment and execute the cells sequentially.

### Dependencies

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- h5py

## Author

**Barbod Alamian**  
[GitHub Profile](https://github.com/BarbodAlamian)

## License

This project is developed for educational and research purposes as part of a university assignment. Feel free to use it for learning and experimentation.
