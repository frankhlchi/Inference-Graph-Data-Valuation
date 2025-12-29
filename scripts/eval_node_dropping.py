#!/usr/bin/env python3
"""
Run node-dropping evaluation and compute AUC from an existing run directory.

Uses already-sampled permutations in:
  <run_dir>/<Dataset>/seed_<seed>/test_samples/
and the saved SVGL weights in:
  <run_dir>/<Dataset>/seed_<seed>/result.json (selected_features).
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch

# Add repo root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svgl.data.datasets import load_dataset
from svgl.data.preprocess import load_preprocessed_data
from svgl.evaluation.node_dropping import (
    estimate_shapley_from_samples,
    node_dropping_curve,
    parse_selected_features,
)
from svgl.models.gnn import create_model
from svgl.utils import fix_seed


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "data")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate_one(run_dir: Path, dataset: str, seed: int, device: str, stride: int) -> dict:
    exp_dir = run_dir / dataset / f"seed_{seed}"
    result_path = exp_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing {result_path}")

    res = _load_json(result_path)
    cfg = res.get("config") or {}
    if not cfg:
        raise ValueError(f"Missing config in {result_path}")

    weights = parse_selected_features(res.get("selected_features"))
    if not weights:
        raise ValueError(f"No selected_features/weights in {result_path}")

    cache_dir = exp_dir / "cache"
    split_data = load_preprocessed_data(
        dataset,
        cache_dir=str(cache_dir),
        setting=cfg.get("setting", "inductive"),
        use_pmlp=bool(cfg.get("use_pmlp", True)),
        data_seed=seed,
    )

    fix_seed(seed)
    ds = load_dataset(dataset, root=DATA_ROOT)
    data = ds[0].to(device)

    # Masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[torch.tensor(split_data["train_indices"], device=device)] = True
    data.train_mask = train_mask

    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask[torch.tensor(split_data["val_indices"], device=device)] = True
    data.val_mask = val_mask

    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask[torch.tensor(split_data["test_indices"], device=device)] = True
    data.test_mask = test_mask

    # Train base model (same procedure as run_parallel_experiments.py).
    model_cfg = {"hidden_dim": int(cfg["hidden_dim"]), "num_layers": 2, "dropout": 0.5}
    model = create_model(cfg["model"], data.num_features, ds.num_classes, model_cfg, seed=seed).to(device)

    train_data = data.clone()
    train_data.edge_index = torch.tensor(split_data["train_edge_index"], dtype=torch.long, device=device)
    model.fit(
        train_data,
        num_epochs=int(cfg["num_epochs"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        device=device,
    )

    test_edge_index = torch.tensor(split_data["test_edge_index"], dtype=torch.long, device=device)
    eval_data = data.clone()
    eval_data.edge_index = test_edge_index

    # Estimate test-time Shapley values from already sampled permutations.
    test_samples_dir = exp_dir / "test_samples"
    shap_pred, shap_true, n_samples = estimate_shapley_from_samples(
        str(test_samples_dir),
        weights=weights,
        utility_key="test_acc",
    )

    ranked_nodes = [n for n, _ in sorted(shap_pred.items(), key=lambda kv: kv[1], reverse=True)]
    K = len(ranked_nodes)

    ks, accs = node_dropping_curve(
        model=model,
        data=eval_data,
        base_edge_index=test_edge_index,
        ranked_nodes=ranked_nodes,
        device=device,
        stride=stride,
    )

    # AUC in paper is effectively the (discrete) area under the accuracy curve.
    # With stride==1 and ks==[0..K], auc_sum = sum_{k=1..K} Acc_k.
    if stride == 1 and ks and ks[0] == 0 and ks[-1] == K and len(ks) == K + 1:
        auc_sum = float(sum(accs[1:]))
    else:
        auc_sum = 0.0
        for i in range(len(ks) - 1):
            auc_sum += 0.5 * (accs[i] + accs[i + 1]) * (ks[i + 1] - ks[i])
        auc_sum = float(auc_sum)

    auc_mean = float(auc_sum / K) if K > 0 else float("nan")

    return {
        "dataset": dataset,
        "seed": seed,
        "device": device,
        "stride": stride,
        "K": K,
        "num_perm_samples": n_samples,
        "auc_sum": auc_sum,
        "auc_mean": auc_mean,
        "acc_curve": {"k": ks, "acc": accs},
        "top_nodes": ranked_nodes[:50],
    }


def _mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.pstdev(xs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "node_dropping"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            print(f"[node-dropping] {ds} seed={seed} ...")
            row = _evaluate_one(run_dir, ds, seed, args.device, args.stride)
            all_rows.append(row)
            out_path = out_dir / f"{ds}_seed{seed}.json"
            out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"  wrote {out_path}")

    # Aggregate by dataset
    summary = {}
    for ds in args.datasets:
        rows = [r for r in all_rows if r["dataset"] == ds]
        aucs = [r["auc_sum"] for r in rows]
        m, s = _mean_std(aucs)
        summary[ds] = {
            "num_runs": len(rows),
            "auc_sum_mean": m,
            "auc_sum_std": s,
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")

    # Markdown table
    lines = []
    lines.append("# Node Dropping AUC Summary")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- device: `{args.device}`")
    lines.append(f"- stride: `{args.stride}`")
    lines.append("")
    lines.append("| Dataset | Runs | AUC(sum) mean±std |")
    lines.append("|---|---:|---:|")
    for ds in args.datasets:
        s = summary.get(ds, {})
        if s.get("auc_sum_mean") is None:
            lines.append(f"| {ds} | {s.get('num_runs',0)} | - |")
        else:
            lines.append(f"| {ds} | {s['num_runs']} | {s['auc_sum_mean']:.2f}±{s['auc_sum_std']:.2f} |")
    md_path = out_dir / "SUMMARY.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
