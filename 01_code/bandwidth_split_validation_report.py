import argparse
import os
import time
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from activation_corruption import (
    drop_packets_preserve_order,
    get_total_packets,
    reconstruct_from_observed_packets,
    sample_missing_positions,
)
from models import SUPPORTED_MODELS, get_num_splits, load_model
from splitter import SplitModel


IMAGENETTE_WNID_TO_IDX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-only report across bandwidth-dependent optimal split layers"
    )
    parser.add_argument("--val-dir", required=True, help="ImageFolder validation directory")
    parser.add_argument("--dataset-name", default="auto", choices=["auto", "imagenet", "imagenette"])
    parser.add_argument("--model-name", default="vit_b_16", choices=SUPPORTED_MODELS)
    parser.add_argument(
        "--bandwidths-mbps",
        default="1,2,5,10,20,50,100",
        help="Comma-separated bandwidth values in MB/s",
    )
    parser.add_argument("--packet-elems", type=int, default=256)
    parser.add_argument("--missing-packets", type=int, default=20)
    parser.add_argument(
        "--missing-packets-levels",
        default=None,
        help="Comma-separated packet-loss levels to evaluate. Overrides --missing-packets.",
    )
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--out-dir", default="bandwidth_split_validation_report")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return "cpu"
    return device_arg


def parse_bandwidths(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_missing_levels(args) -> List[int]:
    if args.missing_packets_levels:
        return [int(x.strip()) for x in args.missing_packets_levels.split(",") if x.strip()]
    return [args.missing_packets]


def infer_dataset_name(val_dir: str, dataset_name: str) -> str:
    if dataset_name != "auto":
        return dataset_name
    return "imagenette" if "imagenette" in val_dir.lower() else "imagenet"


def resolve_dataset_root(val_dir: str) -> str:
    if os.path.isdir(os.path.join(val_dir, "val")):
        return os.path.join(val_dir, "val")
    return val_dir


def build_target_remap(classes: List[str], dataset_name: str) -> torch.Tensor:
    if dataset_name == "imagenet":
        return torch.tensor(list(range(len(classes))), dtype=torch.long)
    if dataset_name == "imagenette":
        missing = [c for c in classes if c not in IMAGENETTE_WNID_TO_IDX]
        if missing:
            raise ValueError(f"Unsupported Imagenette classes: {missing}")
        return torch.tensor([IMAGENETTE_WNID_TO_IDX[c] for c in classes], dtype=torch.long)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_loader(val_dir: str, max_samples: int, dataset_name: str):
    val_dir = resolve_dataset_root(val_dir)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    target_remap = build_target_remap(dataset.classes, dataset_name)
    if max_samples > 0 and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return loader, target_remap


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    topk = torch.topk(logits, k=k, dim=1).indices
    return int((topk == target.unsqueeze(1)).any(dim=1).sum().item())


def measure_split_timings(model, device: str, timing_runs: int):
    num_splits = get_num_splits(model)
    x = torch.randn(1, 3, 224, 224, device=device)
    rows = []

    for split_idx in range(num_splits + 1):
        split = SplitModel(model, split_idx)
        edge_ms_vals = []
        server_ms_vals = []
        activation_mb = None

        for _ in range(timing_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            act = split.edge_forward(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            _ = split.server_forward(act)
            if device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            edge_ms_vals.append((t1 - t0) * 1000.0)
            server_ms_vals.append((t2 - t1) * 1000.0)
            activation_mb = act.numel() * 4.0 / (1024.0**2)

        rows.append(
            {
                "split_idx": split_idx,
                "edge_ms": sum(edge_ms_vals) / len(edge_ms_vals),
                "server_ms": sum(server_ms_vals) / len(server_ms_vals),
                "activation_mb": activation_mb,
            }
        )

    return pd.DataFrame(rows)


def compute_optimal_splits(df_splits: pd.DataFrame, bandwidths_mbps: List[float]):
    rows = []
    for bw in bandwidths_mbps:
        df = df_splits.copy()
        df["bandwidth_mbps"] = bw
        df["comm_ms"] = (df["activation_mb"] / bw) * 1000.0
        df["total_ms"] = df["edge_ms"] + df["server_ms"] + df["comm_ms"]
        best = df.loc[df["total_ms"].idxmin()]
        rows.append(
            {
                "bandwidth_mbps": bw,
                "optimal_split_idx": int(best["split_idx"]),
                "edge_ms": float(best["edge_ms"]),
                "server_ms": float(best["server_ms"]),
                "comm_ms": float(best["comm_ms"]),
                "total_ms": float(best["total_ms"]),
                "activation_mb": float(best["activation_mb"]),
            }
        )
    return pd.DataFrame(rows)


def init_metrics():
    return {"total": 0, "loss": 0.0, "top1": 0, "top5": 0}


def update_metrics(metrics, logits: torch.Tensor, target: torch.Tensor):
    loss = F.cross_entropy(logits, target)
    batch = target.shape[0]
    metrics["total"] += batch
    metrics["loss"] += float(loss.item()) * batch
    metrics["top1"] += topk_correct(logits, target, 1)
    metrics["top5"] += topk_correct(logits, target, 5)


def finalize_metrics(metrics, mode: str, bandwidth_mbps: float, split_idx: int, total_packets: int, missing_packets: int, sample_count: int):
    total = metrics["total"]
    return {
        "bandwidth_mbps": bandwidth_mbps,
        "split_idx": split_idx,
        "mode": mode,
        "total_packets": total_packets,
        "missing_packets": missing_packets,
        "sample_count": sample_count,
        "top1_acc": 100.0 * metrics["top1"] / total,
        "top5_acc": 100.0 * metrics["top5"] / total,
        "avg_ce_loss": metrics["loss"] / total,
        "samples": total,
    }


def evaluate_modes_for_split(split, loader, target_remap, packet_elems: int, missing_packets: int, sample_count: int, seed: int, device: str):
    clean_metrics = init_metrics()
    oracle_metrics = init_metrics()
    random_metrics = init_metrics()
    total_packets = None

    with torch.no_grad():
        for sample_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = target_remap[y].to(device)

            clean_act = split.edge_forward(x)
            if total_packets is None:
                total_packets = get_total_packets(clean_act, packet_elems)
            missing_packets_eff = min(missing_packets, total_packets)
            oracle_missing = sample_missing_positions(total_packets, missing_packets_eff, seed + sample_idx)
            observed_packets = drop_packets_preserve_order(clean_act.cpu(), packet_elems, oracle_missing)

            clean_logits = split.server_forward(clean_act)
            update_metrics(clean_metrics, clean_logits, y)

            oracle_act = reconstruct_from_observed_packets(
                observed_packets,
                original_shape=tuple(clean_act.shape),
                packet_elems=packet_elems,
                missing_positions=oracle_missing,
                dtype=clean_act.dtype,
                device=device,
            )
            oracle_logits = split.server_forward(oracle_act)
            update_metrics(oracle_metrics, oracle_logits, y)

            for hypothesis_idx in range(sample_count):
                candidate_missing = sample_missing_positions(
                    total_packets,
                    missing_packets_eff,
                    seed + 100000 + sample_idx * sample_count + hypothesis_idx,
                )
                candidate_act = reconstruct_from_observed_packets(
                    observed_packets,
                    original_shape=tuple(clean_act.shape),
                    packet_elems=packet_elems,
                    missing_positions=candidate_missing,
                    dtype=clean_act.dtype,
                    device=device,
                )
                candidate_logits = split.server_forward(candidate_act)
                update_metrics(random_metrics, candidate_logits, y)

    return (
        total_packets,
        missing_packets_eff,
        clean_metrics,
        oracle_metrics,
        random_metrics,
    )


def plot_latency(df_optimal: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(df_optimal["bandwidth_mbps"], df_optimal["optimal_split_idx"], marker="o", linewidth=2)
    ax[0].set_title("Optimal Split vs Bandwidth")
    ax[0].set_xlabel("Bandwidth (MB/s)")
    ax[0].set_ylabel("Optimal Split Index")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df_optimal["bandwidth_mbps"], df_optimal["edge_ms"], label="edge")
    ax[1].plot(df_optimal["bandwidth_mbps"], df_optimal["server_ms"], label="server")
    ax[1].plot(df_optimal["bandwidth_mbps"], df_optimal["comm_ms"], label="comm")
    ax[1].plot(df_optimal["bandwidth_mbps"], df_optimal["total_ms"], label="total", linewidth=3)
    ax[1].set_title("Latency Components at Optimal Split")
    ax[1].set_xlabel("Bandwidth (MB/s)")
    ax[1].set_ylabel("Latency (ms)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "optimal_split_latency.png"), dpi=160)
    plt.close(fig)


def plot_validation(df_eval: pd.DataFrame, out_dir: str):
    modes = ["clean", "oracle", "random"]
    colors = {"clean": "green", "oracle": "crimson", "random": "tab:blue"}
    missing_levels = sorted(df_eval["missing_packets"].unique())

    for missing_packets in missing_levels:
        df_loss = df_eval[df_eval["missing_packets"] == missing_packets]
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        for mode in modes:
            df_mode = df_loss[df_loss["mode"] == mode].sort_values("bandwidth_mbps")
            ax[0].plot(df_mode["bandwidth_mbps"], df_mode["top1_acc"], marker="o", label=mode, color=colors[mode])
            ax[1].plot(df_mode["bandwidth_mbps"], df_mode["top5_acc"], marker="o", label=mode, color=colors[mode])
            ax[2].plot(df_mode["bandwidth_mbps"], df_mode["avg_ce_loss"], marker="o", label=mode, color=colors[mode])

        ax[0].set_title(f"Top-1 Accuracy (k={missing_packets})")
        ax[1].set_title(f"Top-5 Accuracy (k={missing_packets})")
        ax[2].set_title(f"Average CE Loss (k={missing_packets})")
        ax[0].set_ylabel("%")
        ax[2].set_ylabel("Loss")

        for a in ax:
            a.set_xlabel("Bandwidth (MB/s)")
            a.grid(True, alpha=0.3)
            a.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"validation_modes_vs_bandwidth_k{missing_packets}.png"),
            dpi=160,
        )
        plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    dataset_name = infer_dataset_name(args.val_dir, args.dataset_name)
    loader, target_remap = build_loader(args.val_dir, args.max_samples, dataset_name)
    missing_levels = parse_missing_levels(args)

    model = load_model(model_name=args.model_name, device=device)
    df_splits = measure_split_timings(model, device=device, timing_runs=args.timing_runs)
    df_splits.to_csv(os.path.join(args.out_dir, "split_timing_profile.csv"), index=False)

    bandwidths = parse_bandwidths(args.bandwidths_mbps)
    df_optimal = compute_optimal_splits(df_splits, bandwidths)
    df_optimal.to_csv(os.path.join(args.out_dir, "optimal_splits_by_bandwidth.csv"), index=False)

    eval_rows = []
    unique_splits_done = {}

    for _, row in df_optimal.iterrows():
        bw = float(row["bandwidth_mbps"])
        split_idx = int(row["optimal_split_idx"])

        for missing_level in missing_levels:
            cache_key = (split_idx, missing_level)
            if cache_key not in unique_splits_done:
                split = SplitModel(model, split_idx)
                total_packets, missing_packets_eff, clean_metrics, oracle_metrics, random_metrics = evaluate_modes_for_split(
                    split,
                    loader,
                    target_remap,
                    packet_elems=args.packet_elems,
                    missing_packets=missing_level,
                    sample_count=args.sample_count,
                    seed=args.seed + split_idx * 1000 + missing_level * 100,
                    device=device,
                )
                unique_splits_done[cache_key] = {
                    "total_packets": total_packets,
                    "missing_packets": missing_packets_eff,
                    "clean": clean_metrics,
                    "oracle": oracle_metrics,
                    "random": random_metrics,
                }

            cached = unique_splits_done[cache_key]
            eval_rows.extend(
                [
                    finalize_metrics(cached["clean"], "clean", bw, split_idx, cached["total_packets"], cached["missing_packets"], 1),
                    finalize_metrics(cached["oracle"], "oracle", bw, split_idx, cached["total_packets"], cached["missing_packets"], 1),
                    finalize_metrics(cached["random"], "random", bw, split_idx, cached["total_packets"], cached["missing_packets"], args.sample_count),
                ]
            )

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(os.path.join(args.out_dir, "validation_report_by_bandwidth.csv"), index=False)

    plot_latency(df_optimal, args.out_dir)
    plot_validation(df_eval, args.out_dir)

    df_report = df_eval.merge(
        df_optimal,
        left_on=["bandwidth_mbps", "split_idx"],
        right_on=["bandwidth_mbps", "optimal_split_idx"],
        how="left",
    )
    df_report.to_csv(os.path.join(args.out_dir, "bandwidth_split_report.csv"), index=False)

    print(f"Model used: {args.model_name}")
    print("Optimal split summary:")
    for _, row in df_optimal.iterrows():
        print(
            f"bw={row['bandwidth_mbps']:.2f}MB/s split={int(row['optimal_split_idx'])} "
            f"edge={row['edge_ms']:.2f}ms server={row['server_ms']:.2f}ms "
            f"comm={row['comm_ms']:.2f}ms total={row['total_ms']:.2f}ms"
        )
    print(f"Dataset mapping used: {dataset_name}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
