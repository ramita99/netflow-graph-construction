# NetFlow graph dataset construction

Graph preparation code for *Structural Contribution Analysis of an Edge-Centric GNN for NetFlow-Based Intrusion Detection under Feature Constraints*.

## Install

Use Python 3.12. The standalone preparation code was checked with PyTorch 2.8.0 and PyTorch Geometric 2.6.1:

```bash
python -m pip install torch torch-geometric numpy pandas scikit-learn joblib
```

## Prepare the graphs

Download the original **v3** CSV files from the [University of Queensland NetFlow dataset page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/). Keep the original CSV row order. Supply your own local input and output paths:

```bash
python build_graph_dataset.py --dataset unsw --csv /path/to/NF-UNSW-NB15-v3.csv --out ./graphs/unsw
python build_graph_dataset.py --dataset cic --csv /path/to/NF-CSE-CIC-IDS2018-v3.csv --out ./graphs/cic
python build_graph_dataset.py --dataset ton --csv /path/to/NF-ToN-IoT-v3.csv --out ./graphs/ton
```

Each output directory must be empty. The settings used in the paper are included in the code:

| Dataset option | Records per window | Minimum retained edges | Graphs: train / validation / test |
|---|---:|---:|---|
| `unsw` | 50,000 | 50,000 | 28 / 9 / 10 |
| `cic` | 100,000 | 50,000 | 120 / 40 / 41 |
| `ton` | 100,000 | 50,000 | 165 / 55 / 55 |

Windows follow consecutive source records. Each flow becomes a directed edge between its endpoint IPs; parallel edges and self-loops are retained. IP addresses, labels, attack annotations, and absolute timestamps are excluded from record features. Ports and protocol use fixed one-hot encodings. The 60/20/20 graph-window split is generated once using a fixed split seed of 7 and stratifies on attack presence; the resulting assignment is reused unchanged across all models, feature views, and training seeds. The five reported seeds (3, 7, 11, 13, and 17) are training seeds and do not alter the data split. Numerical scaling statistics are fitted on training windows only.

## Load a split and feature view

```python
from build_graph_dataset import GraphDataset

dataset = GraphDataset("./graphs/unsw", split="train", view="all")
graph = dataset[0]
print(graph.edge_index.shape, graph.edge_attr.shape, graph.edge_y.shape)
```

Splits: `train`, `val`, `test`. Views: `all`, `ports_proto`, `ports_only`, `proto_only`, `featureless`.

The generated `windows_raw/*.pt` files contain unscaled PyTorch Geometric graphs. `GraphDataset` applies the training-fitted scaler when loading `all`, selects the matching columns for restricted views, or supplies constant ones for `featureless`. Binary edge labels are stored in `edge_y`. The generated `meta_ext.json` records feature names, source-row ranges, and split assignments; `scaler_random.joblib` stores the numerical scaler.
