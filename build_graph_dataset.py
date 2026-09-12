"""Construct the NetFlow-v3 graph inputs used in the HCIS study.

Graph construction, feature preprocessing, and graph-level data splitting only.
Run ``python build_graph_dataset.py --help`` for the preparation command.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


# Settings used for the three NetFlow-v3 datasets.
DATASETS = {
    "unsw": {
        "dataset_key": "nf_unsw_nb15_v3", "name": "NF-UNSW-NB15-v3",
        "window_size": 50000, "chunksize": 50000, "min_edges": 50000,
        "expected_graphs": 47, "expected_split_counts": {"train": 28, "val": 9, "test": 10},
    },
    "cic": {
        "dataset_key": "nf_cse_cic_ids2018_v3", "name": "NF-CSE-CIC-IDS2018-v3",
        "window_size": 100000, "chunksize": 100000, "min_edges": 50000,
        "expected_graphs": 201, "expected_split_counts": {"train": 120, "val": 40, "test": 41},
    },
    "ton": {
        "dataset_key": "nf_ton_iot_v3", "name": "NF-ToN-IoT-v3",
        "window_size": 100000, "chunksize": 100000, "min_edges": 50000,
        "expected_graphs": 275, "expected_split_counts": {"train": 165, "val": 55, "test": 55},
    },
}
SPLIT_SEED = 7
SPLIT_RATIO = (0.6, 0.2, 0.2)
SPLIT_PROFILE = "random_graph_60_20_20"
CLIP_VALUE = 1.0e9
TOP_PORTS = [
    20, 21, 22, 23, 25, 53, 80, 110, 111, 123, 135, 137, 138, 139, 143, 161,
    389, 443, 445, 465, 514, 587, 636, 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 8080,
]
PORT_OH_DIM = 2 + len(TOP_PORTS)
RESERVED_EXCLUDE_CANDIDATES = {
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "SRC_IP", "DST_IP", "SOURCE_IP",
    "DESTINATION_IP", "LABEL", "BINARY_LABEL", "ATTACK", "ATTACK_TYPE",
    "ATTACK_CATEGORY", "CLASS", "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS", "FLOW_START_TIME", "FLOW_END_TIME", "TIMESTAMP",
}
TIMESTAMP_PREFERRED = ["FLOW_START_MILLISECONDS"]
TIMESTAMP_FALLBACK = ["FLOW_END_MILLISECONDS", "FLOW_START_TIME", "FLOW_END_TIME", "TIMESTAMP"]
ATTACK_CANDIDATES = ["ATTACK", "ATTACK_TYPE", "ATTACK_CATEGORY"]
FEATURE_VIEWS = ("all", "ports_proto", "ports_only", "proto_only", "featureless")



@dataclass
class Schema:
    src_ip: str
    dst_ip: str
    label: str


def _normalize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()


def find_col(columns: list[str], candidates: Iterable[str]) -> str | None:
    normalized = {_normalize_name(column): column for column in columns}
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def find_ports_proto_cols(columns: list[str]) -> tuple[str | None, str | None, str | None]:
    src_port = find_col(columns, ["L4_SRC_PORT", "SRC_PORT", "SPORT", "SOURCE_PORT", "SRCPORT", "TCP_SRC_PORT", "UDP_SRC_PORT"])
    dst_port = find_col(columns, ["L4_DST_PORT", "DST_PORT", "DPORT", "DEST_PORT", "DSTPORT", "TCP_DST_PORT", "UDP_DST_PORT"])
    proto = find_col(columns, ["PROTOCOL", "PROTO", "L4_PROTO", "IP_PROTO", "IPPROTOCOL"])
    return src_port, dst_port, proto


def iter_csv_chunks(csv_path: str, chunksize: int, usecols: list[str] | None = None):
    yield from pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=usecols,
        low_memory=True,
        engine="c",
    )


def sanitize_numeric(values, clip: float = 1e9) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array[~np.isfinite(array)] = np.nan
    np.clip(array, -clip, clip, out=array)
    return np.nan_to_num(array, nan=0.0, posinf=clip, neginf=-clip)


def encode_port_top_onehot(values: np.ndarray) -> np.ndarray:
    ports = np.asarray(values, dtype=np.float64)
    num_edges = ports.shape[0]
    bucket = np.ones(num_edges, dtype=np.int64)

    valid = np.isfinite(ports) & (ports >= 0) & (ports <= 65535)
    bucket[~valid] = 0

    port_int = np.zeros(num_edges, dtype=np.int64)
    port_int[valid] = ports[valid].astype(np.int64)
    for idx, port in enumerate(TOP_PORTS):
        bucket[valid & (port_int == port)] = 2 + idx

    encoded = np.zeros((num_edges, PORT_OH_DIM), dtype=np.float32)
    encoded[np.arange(num_edges), bucket] = 1.0
    return encoded


def encode_proto_onehot(values: np.ndarray) -> np.ndarray:
    proto = np.asarray(values, dtype=np.float64)
    num_edges = proto.shape[0]
    bucket = np.zeros(num_edges, dtype=np.int64)

    valid = np.isfinite(proto) & (proto >= 0) & (proto <= 255)
    proto_int = np.zeros(num_edges, dtype=np.int64)
    proto_int[valid] = proto[valid].astype(np.int64)

    bucket[valid] = 4
    bucket[valid & (proto_int == 1)] = 1
    bucket[valid & (proto_int == 6)] = 2
    bucket[valid & (proto_int == 17)] = 3

    encoded = np.zeros((num_edges, 5), dtype=np.float32)
    encoded[np.arange(num_edges), bucket] = 1.0
    return encoded


def parse_binary_labels(
    series: pd.Series,
    pos_labels: list[str] | None = None,
    neg_labels: list[str] | None = None,
) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return (numeric.fillna(0) > 0).astype(np.float32).to_numpy()

    labels = series.astype(str).str.strip().str.lower()
    pos_set = {value.strip().lower() for value in (pos_labels or [])}
    neg_set = {value.strip().lower() for value in (neg_labels or [])}

    if pos_set or neg_set:
        encoded = np.zeros(len(labels), dtype=np.float32)
        if pos_set:
            encoded[labels.isin(pos_set).to_numpy()] = 1.0
        if neg_set:
            encoded[labels.isin(neg_set).to_numpy()] = 0.0
        return encoded

    benign_aliases = {"benign", "normal", "background", "legit", "legitimate", "clean"}
    if labels.isin(benign_aliases).any():
        return (~labels.isin(benign_aliases)).astype(np.float32).to_numpy()

    unique_values = sorted(labels.unique().tolist())
    if len(unique_values) == 2:
        return (labels == unique_values[1]).astype(np.float32).to_numpy()

    raise ValueError("Could not parse the binary labels. Use the binary Label field of the NetFlow dataset.")


def build_window_graph(
    df_window: pd.DataFrame,
    schema: Schema,
    feature_cols: list[str],
    feature_set: str,
    clip_value: float,
    pos_labels: list[str] | None = None,
    neg_labels: list[str] | None = None,
) -> Data:
    src = df_window[schema.src_ip].astype(str).to_numpy()
    dst = df_window[schema.dst_ip].astype(str).to_numpy()

    nodes = pd.Index(np.unique(np.concatenate([src, dst])))
    node_to_id = {node: idx for idx, node in enumerate(nodes.tolist())}
    src_id = np.fromiter((node_to_id[value] for value in src), dtype=np.int64, count=len(src))
    dst_id = np.fromiter((node_to_id[value] for value in dst), dtype=np.int64, count=len(dst))
    edge_index = torch.tensor(np.vstack([src_id, dst_id]), dtype=torch.long)

    if feature_set == "featureless" or not feature_cols:
        edge_attr = torch.ones((len(df_window), 1), dtype=torch.float32)
    elif feature_set == "ports_only":
        src_port, dst_port, _ = find_ports_proto_cols(list(df_window.columns))
        if src_port is None or dst_port is None:
            raise ValueError("ports_only requires source and destination port columns.")
        src_values = pd.to_numeric(df_window[src_port], errors="coerce").to_numpy()
        dst_values = pd.to_numeric(df_window[dst_port], errors="coerce").to_numpy()
        edge_attr = torch.from_numpy(
            np.concatenate([encode_port_top_onehot(src_values), encode_port_top_onehot(dst_values)], axis=1).astype(np.float32)
        )
    elif feature_set == "proto_only":
        _, _, proto = find_ports_proto_cols(list(df_window.columns))
        if proto is None:
            raise ValueError("proto_only requires a protocol column.")
        proto_values = pd.to_numeric(df_window[proto], errors="coerce").to_numpy()
        edge_attr = torch.from_numpy(encode_proto_onehot(proto_values).astype(np.float32))
    elif feature_set == "ports_proto":
        src_port, dst_port, proto = find_ports_proto_cols(
            list(df_window.columns))
        if src_port is None or dst_port is None \
                or proto is None:
            raise ValueError("ports_proto requires source/destination ports and protocol.")
        # Use the same port encoding in every view.
        src_values = pd.to_numeric(
            df_window[src_port], errors="coerce"
        ).to_numpy()
        dst_values = pd.to_numeric(
            df_window[dst_port], errors="coerce"
        ).to_numpy()
        proto_values = pd.to_numeric(
            df_window[proto], errors="coerce"
        ).to_numpy()
        edge_attr = torch.from_numpy(
            np.concatenate([
                encode_port_top_onehot(src_values),  # 34d
                encode_port_top_onehot(dst_values),  # 34d
                encode_proto_onehot(proto_values),   # 5d
            ], axis=1).astype(np.float32)
        )
        # 73 dimensions: 34 + 34 + 5.
    else:  # feature_set == "all"
        # One-hot ports/protocol, followed by numerical fields.
        src_port, dst_port, proto = find_ports_proto_cols(
            list(df_window.columns))
        port_proto_cols = {
            c for c in (src_port, dst_port, proto)
            if c is not None
        }
        other_cols = [
            c for c in feature_cols
            if c not in port_proto_cols
        ]
        parts = []
        if src_port and src_port in feature_cols:
            sv = pd.to_numeric(df_window[src_port],
                            errors="coerce").to_numpy()
            parts.append(encode_port_top_onehot(sv))  # 34d
        if dst_port and dst_port in feature_cols:
            dv = pd.to_numeric(df_window[dst_port],
                            errors="coerce").to_numpy()
            parts.append(encode_port_top_onehot(dv))  # 34d
        if proto and proto in feature_cols:
            pv = pd.to_numeric(df_window[proto],
                            errors="coerce").to_numpy()
            parts.append(encode_proto_onehot(pv))     # 5d
        if other_cols:
            num = df_window[other_cols].apply(
                pd.to_numeric, errors="coerce")
            parts.append(sanitize_numeric(
                num.to_numpy(), clip=clip_value
            ).astype(np.float32))
        edge_attr = torch.from_numpy(
            np.concatenate(parts, axis=1).astype(np.float32)
        )

    edge_y = torch.from_numpy(parse_binary_labels(df_window[schema.label], pos_labels, neg_labels))
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
    data.edge_y = edge_y
    return data


def split_windows_stratified(pos_edges: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    window_count = len(pos_edges)
    indices = np.arange(window_count)
    has_positive = (pos_edges > 0).astype(int)

    pos_idx = indices[has_positive == 1].tolist()
    neg_idx = indices[has_positive == 0].tolist()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    train_count = int(window_count * train_ratio)
    val_count = int(window_count * val_ratio)
    test_count = window_count - train_count - val_count

    pos_total = len(pos_idx)
    pos_train = int(round(pos_total * train_ratio))
    pos_val = int(round(pos_total * val_ratio))
    pos_test = pos_total - pos_train - pos_val

    if pos_total >= 3:
        if pos_val <= 0:
            pos_val = 1
            pos_train = max(1, pos_train - 1)
        if pos_test <= 0:
            pos_test = 1
            pos_train = max(1, pos_train - 1)
        pos_train = min(pos_train, pos_total - pos_val - pos_test)

    pos_train = min(pos_train, train_count)
    pos_val = min(pos_val, val_count)
    pos_test = min(pos_test, test_count)

    train_pos = pos_idx[:pos_train]
    val_pos = pos_idx[pos_train:pos_train + pos_val]
    test_pos = pos_idx[pos_train + pos_val:pos_train + pos_val + pos_test]
    train_pos.extend(pos_idx[pos_train + pos_val + pos_test:])

    neg_train = neg_idx[:max(0, train_count - len(train_pos))]
    neg_val_start = len(neg_train)
    neg_val = neg_idx[neg_val_start:neg_val_start + max(0, val_count - len(val_pos))]
    neg_test_start = neg_val_start + len(neg_val)
    neg_test = neg_idx[neg_test_start:neg_test_start + max(0, test_count - len(test_pos))]

    train_idx = train_pos + neg_train
    val_idx = val_pos + neg_val
    test_idx = test_pos + neg_test
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def resolve_columns(columns: list[str]) -> dict[str, Any]:
    """Normalize header, locate schema / attack / timestamp cols, reserved exclusions."""
    norm_reserved = {_normalize_name(c) for c in RESERVED_EXCLUDE_CANDIDATES}
    src_ip = find_col(columns, ["IPV4_SRC_ADDR", "SRC_IP", "SRCIP", "SOURCEIP", "SRCADDR", "SADDR"])
    dst_ip = find_col(columns, ["IPV4_DST_ADDR", "DST_IP", "DSTIP", "DESTIP", "DESTINATIONIP", "DSTADDR", "DADDR"])
    label = find_col(columns, ["Label", "LABEL", "BINARY_LABEL", "CLASS", "y"])
    attack = find_col(columns, ATTACK_CANDIDATES)
    ts_col, ts_source = None, None
    for cand in TIMESTAMP_PREFERRED:
        ts_col = find_col(columns, [cand])
        if ts_col:
            ts_source = "preferred"
            break
    if ts_col is None:
        for cand in TIMESTAMP_FALLBACK:
            ts_col = find_col(columns, [cand])
            if ts_col:
                ts_source = "fallback"
                break
    if src_ip is None or dst_ip is None or label is None:
        raise ValueError("Could not resolve src/dst IP or label columns from header.")

    excluded: dict[str, str] = {src_ip: "endpoint_identifier", dst_ip: "endpoint_identifier",
                               label: "label"}
    if attack:
        excluded[attack] = "attack_annotation"
    for col in columns:
        if _normalize_name(col) in norm_reserved and col not in excluded:
            excluded[col] = "reserved_exclude_candidate"
    if ts_col and ts_col not in excluded:
        excluded[ts_col] = "absolute_timestamp"
    return {
        "src_ip": src_ip, "dst_ip": dst_ip, "label": label,
        "attack_col": attack, "timestamp_col": ts_col, "timestamp_source": ts_source,
        "excluded": excluded,
        "feature_cols": [c for c in columns if c not in excluded],
    }


def _feature_names_for_all(preview_columns: list[str], feature_cols: list[str]) -> tuple[list[str], list[int], list[int], dict[str, str]]:
    """Column names for the mixed one-hot/numerical ALL representation.

    Returns (names, numeric_idx, ohe_idx, name->source_col)."""
    src_port, dst_port, proto = find_ports_proto_cols(preview_columns)
    port_proto = {c for c in (src_port, dst_port, proto) if c is not None}
    other_cols = [c for c in feature_cols if c not in port_proto]
    names: list[str] = []
    source: dict[str, str] = {}
    if src_port and src_port in feature_cols:
        block = ["src_port__MISSING", "src_port__OTHER"] + [f"src_port__{p}" for p in TOP_PORTS]
        names += block
        source.update({n: src_port for n in block})
    if dst_port and dst_port in feature_cols:
        block = ["dst_port__MISSING", "dst_port__OTHER"] + [f"dst_port__{p}" for p in TOP_PORTS]
        names += block
        source.update({n: dst_port for n in block})
    if proto and proto in feature_cols:
        block = ["proto__MISSING", "proto__ICMP", "proto__TCP", "proto__UDP", "proto__OTHER"]
        names += block
        source.update({n: proto for n in block})
    ohe_idx = list(range(len(names)))
    numeric_idx = list(range(len(names), len(names) + len(other_cols)))
    names += other_cols
    source.update({c: c for c in other_cols})
    return names, numeric_idx, ohe_idx, source


def _load_graph(path: Path) -> Data:
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare_graph_dataset(dataset: str, csv_path: str | Path, output_dir: str | Path) -> dict:
    """Prepare raw window graphs, graph splits, and a training-only numerical scaler.

    Source CSV row order is preserved. Scaling is applied by GraphDataset on load.
    Use an empty output directory for each preparation run.
    """
    reg = DATASETS[dataset]
    csv_path = Path(csv_path).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    window_dir = output_dir / "windows_raw"
    window_dir.mkdir(parents=True)
    window_size, min_edges = reg["window_size"], reg["min_edges"]

    columns = list(pd.read_csv(csv_path, nrows=5000).columns)
    colinfo = resolve_columns(columns)
    schema = Schema(colinfo["src_ip"], colinfo["dst_ip"], colinfo["label"])
    feature_cols = colinfo["feature_cols"]
    if not feature_cols:
        raise ValueError("No features remain after excluding endpoint/annotation fields.")
    names, numeric_idx, ohe_idx, _ = _feature_names_for_all(columns, feature_cols)
    annotation_cols = [c for c in (colinfo["attack_col"], colinfo["timestamp_col"]) if c]
    usecols = list(dict.fromkeys(
        [schema.src_ip, schema.dst_ip, schema.label] + feature_cols + annotation_cols))

    provenance, kept = [], []
    buffer = []
    window_counter = 0
    row_cursor = 0

    def flush_window(frame: pd.DataFrame) -> None:
        nonlocal window_counter, row_cursor
        if frame.empty:
            return
        window_id = window_counter
        window_counter += 1
        start, end = row_cursor, row_cursor + len(frame)
        row_cursor = end
        graph = build_window_graph(frame, schema, feature_cols, "all", CLIP_VALUE)
        count = int(graph.edge_index.size(1))
        positives = float(graph.edge_y.sum().item())
        keep = count >= min_edges
        provenance.append({
            "original_window_id": window_id, "raw_row_start": start,
            "raw_row_end_exclusive": end, "edge_count": count,
            "positive_count": positives, "kept": keep,
        })
        if keep:
            graph.original_window_id = window_id
            graph.raw_row_start = start
            graph.raw_row_end_exclusive = end
            torch.save(graph, window_dir / f"win_{window_id:06d}.pt")
            kept.append({"owid": window_id, "pos": positives})

    for chunk in iter_csv_chunks(str(csv_path), reg["chunksize"], usecols=usecols):
        buffer.append(chunk)
        while sum(len(part) for part in buffer) >= window_size:
            merged = pd.concat(buffer, ignore_index=True)
            flush_window(merged.iloc[:window_size].copy())
            remainder = merged.iloc[window_size:].copy()
            buffer = [remainder] if not remainder.empty else []
    if buffer:
        flush_window(pd.concat(buffer, ignore_index=True))
    if not kept:
        raise ValueError("No windows meet the dataset's minimum edge count.")

    positions = split_windows_stratified(
        np.asarray([rec["pos"] for rec in kept], dtype=np.float64),
        SPLIT_RATIO[0], SPLIT_RATIO[1], SPLIT_SEED)
    assignments = {
        split: sorted(kept[i]["owid"] for i in indices)
        for split, indices in zip(("train", "val", "test"), positions)
    }
    if len(kept) != reg["expected_graphs"]:
        raise ValueError(
            f"Expected {reg['expected_graphs']} graphs for {reg['name']}; got {len(kept)}. "
            "Check the source dataset version and row count.")
    actual_counts = {key: len(values) for key, values in assignments.items()}
    if actual_counts != reg["expected_split_counts"]:
        raise ValueError(f"Unexpected split counts: {actual_counts}")

    if numeric_idx:
        scaler = StandardScaler()
        for window_id in assignments["train"]:
            graph = _load_graph(window_dir / f"win_{window_id:06d}.pt")
            values = sanitize_numeric(graph.edge_attr[:, numeric_idx].numpy(), CLIP_VALUE)
            scaler.partial_fit(values)
        joblib.dump(scaler, output_dir / "scaler_random.joblib")

    meta = {
        "preprocessing_profile": "clean_v2", "dataset_key": reg["dataset_key"],
        "source_csv": csv_path.name,
        "schema": {"src_ip": schema.src_ip, "dst_ip": schema.dst_ip, "label": schema.label},
        "excluded_columns": colinfo["excluded"], "feature_cols_raw": feature_cols,
        "feature_names": names, "numeric_feature_idx": numeric_idx, "ohe_feature_idx": ohe_idx,
        "edge_attr_encoding": "mixed_ohe_ports_proto_num_v1",
        "window_size": window_size, "chunksize": reg["chunksize"], "min_edges": min_edges,
        "n_windows_total": window_counter, "n_windows_kept": len(kept),
        "split_seed": SPLIT_SEED, "split_ratio": list(SPLIT_RATIO),
        "split_assignments": {SPLIT_PROFILE: assignments},
        "scaler_info": {"numeric_dims": len(numeric_idx), "fit_split": "train"},
        "windows": provenance,
    }
    (output_dir / "meta_ext.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class GraphDataset(torch.utils.data.Dataset):
    """A split and feature view over the prepared graph windows."""

    def __init__(self, prepared_dir: str | Path, split: str = "train", view: str = "all"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if view not in FEATURE_VIEWS:
            raise ValueError(f"Unknown view: {view}; choose from {FEATURE_VIEWS}")
        self.root = Path(prepared_dir)
        self.meta = json.loads((self.root / "meta_ext.json").read_text(encoding="utf-8"))
        self.window_ids = self.meta["split_assignments"][SPLIT_PROFILE][split]
        self.view = view
        self.numeric_idx = self.meta["numeric_feature_idx"]
        self.scaler = None
        if view == "all" and self.numeric_idx:
            self.scaler = joblib.load(self.root / "scaler_random.joblib")
        prefixes = {
            "ports_only": ("src_port__", "dst_port__"),
            "proto_only": ("proto__",),
            "ports_proto": ("src_port__", "dst_port__", "proto__"),
        }
        self.feature_idx = [
            i for i, name in enumerate(self.meta["feature_names"])
            if name.startswith(prefixes.get(view, ()))
        ]

    def __len__(self) -> int:
        return len(self.window_ids)

    def __getitem__(self, index: int) -> Data:
        window_id = self.window_ids[index]
        graph = _load_graph(self.root / "windows_raw" / f"win_{window_id:06d}.pt")
        graph.is_featureless = self.view == "featureless"
        if graph.is_featureless:
            graph.edge_attr = torch.ones((int(graph.edge_attr.size(0)), 1), dtype=torch.float32)
        elif self.view != "all":
            graph.edge_attr = graph.edge_attr[:, self.feature_idx]
        elif self.scaler is not None:
            values = graph.edge_attr.numpy().copy()
            values[:, self.numeric_idx] = sanitize_numeric(
                self.scaler.transform(sanitize_numeric(values[:, self.numeric_idx], CLIP_VALUE)),
                CLIP_VALUE)
            graph.edge_attr = torch.from_numpy(values.astype(np.float32))
        return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--csv", required=True, type=Path, help="Original NetFlow-v3 CSV")
    parser.add_argument("--out", required=True, type=Path, help="Empty output directory")
    args = parser.parse_args()
    meta = prepare_graph_dataset(args.dataset, args.csv, args.out)
    sizes = {s: len(ids) for s, ids in meta["split_assignments"][SPLIT_PROFILE].items()}
    print(f"Prepared {meta['n_windows_kept']} graphs in {args.out}: {sizes}")


if __name__ == "__main__":
    main()

