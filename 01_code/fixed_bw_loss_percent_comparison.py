import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from bandwidth_split_validation_report import (
    build_loader,
    compute_optimal_splits,
    evaluate_modes_for_split,
    infer_dataset_name,
    measure_split_timings,
    resolve_device,
)
from models import SUPPORTED_MODELS, load_model
from splitter import SplitModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare models at a fixed bandwidth across packet-loss percentages."
    )
    parser.add_argument("--val-dir", required=True, help="ImageFolder validation directory")
    parser.add_argument("--dataset-name", default="auto", choices=["auto", "imagenet", "imagenette"])
    parser.add_argument(
        "--model-names",
        default="vit_b_16,mobilenet_v2",
        help="Comma-separated model names to compare.",
    )
    parser.add_argument("--bandwidth-mbps", type=float, default=40.0)
    parser.add_argument(
        "--loss-pcts",
        default="0,10,20,30,40,50,60",
        help="Comma-separated packet-loss percentages.",
    )
    parser.add_argument("--packet-elems", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--out-dir", default="fixed_bw_loss_percent_comparison")
    return parser.parse_args()


def parse_csv_list(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def plot_accuracy(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    metric_specs = [("top1_acc", "Top-1 Accuracy (%)"), ("top5_acc", "Top-5 Accuracy (%)")]
    mode_styles = {"clean": "-", "oracle": "--", "random": ":"}

    for ax, (metric_col, title) in zip(axes, metric_specs):
        for (model_name, mode), group in df.groupby(["model_name", "mode"]):
            group = group.sort_values("loss_pct")
            ax.plot(
                group["loss_pct"],
                group[metric_col],
                marker="o",
                linewidth=2.2,
                linestyle=mode_styles.get(mode, "-"),
                label=f"{model_name}:{mode}",
            )
        ax.set_title(title)
        ax.set_xlabel("Packet Loss (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(out_dir, "accuracy_vs_loss_pct.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    dataset_name = infer_dataset_name(args.val_dir, args.dataset_name)
    loader, target_remap = build_loader(args.val_dir, args.max_samples, dataset_name)

    model_names = parse_csv_list(args.model_names)
    unsupported = [m for m in model_names if m not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported models requested: {unsupported}")
    loss_pcts = [float(x) for x in parse_csv_list(args.loss_pcts)]

    split_rows = []
    result_rows = []

    for model_name in model_names:
        model = load_model(model_name).to(device).eval()
        df_splits = measure_split_timings(model, device=device, timing_runs=args.timing_runs)
        optimal_df = compute_optimal_splits(df_splits, [args.bandwidth_mbps])
        best = optimal_df.iloc[0]
        split_idx = int(best["optimal_split_idx"])

        split_rows.append(
            {
                "model_name": model_name,
                "bandwidth_mbps": args.bandwidth_mbps,
                "optimal_split_idx": split_idx,
                "edge_ms": float(best["edge_ms"]),
                "server_ms": float(best["server_ms"]),
                "comm_ms": float(best["comm_ms"]),
                "total_ms": float(best["total_ms"]),
                "activation_mb": float(best["activation_mb"]),
            }
        )

        split = SplitModel(model, split_idx)
        total_packets = None

        for loss_pct in loss_pcts:
            if total_packets is None:
                with torch.no_grad():
                    x0, _ = next(iter(loader))
                    x0 = x0.to(device)
                    act0 = split.edge_forward(x0)
                    total_packets = int((act0.numel() + args.packet_elems - 1) // args.packet_elems)

            missing_packets = int(round(total_packets * loss_pct / 100.0))
            (
                total_packets_eval,
                missing_packets_eff,
                clean_metrics,
                oracle_metrics,
                random_metrics,
            ) = evaluate_modes_for_split(
                split,
                loader,
                target_remap,
                packet_elems=args.packet_elems,
                missing_packets=missing_packets,
                sample_count=args.sample_count,
                seed=args.seed,
                device=device,
            )

            for mode, metrics, sample_count in [
                ("clean", clean_metrics, 1),
                ("oracle", oracle_metrics, 1),
                ("random", random_metrics, args.sample_count),
            ]:
                total = metrics["total"]
                result_rows.append(
                    {
                        "model_name": model_name,
                        "bandwidth_mbps": args.bandwidth_mbps,
                        "optimal_split_idx": split_idx,
                        "loss_pct": loss_pct,
                        "total_packets": total_packets_eval,
                        "missing_packets": missing_packets_eff,
                        "mode": mode,
                        "sample_count": sample_count,
                        "top1_acc": 100.0 * metrics["top1"] / total,
                        "top5_acc": 100.0 * metrics["top5"] / total,
                        "avg_ce_loss": metrics["loss"] / total,
                        "edge_ms": float(best["edge_ms"]),
                        "server_ms": float(best["server_ms"]),
                        "comm_ms": float(best["comm_ms"]),
                        "total_ms": float(best["total_ms"]),
                        "activation_mb": float(best["activation_mb"]),
                    }
                )

    split_df = pd.DataFrame(split_rows)
    result_df = pd.DataFrame(result_rows)

    split_out = os.path.join(args.out_dir, "optimal_splits_fixed_bw.csv")
    result_out = os.path.join(args.out_dir, "fixed_bw_accuracy_comparison.csv")
    table_out = os.path.join(args.out_dir, "fixed_bw_accuracy_table.csv")

    split_df.to_csv(split_out, index=False)
    result_df.to_csv(result_out, index=False)
    result_df[
        [
            "model_name",
            "loss_pct",
            "total_packets",
            "missing_packets",
            "mode",
            "top1_acc",
            "top5_acc",
            "avg_ce_loss",
            "optimal_split_idx",
            "total_ms",
        ]
    ].to_csv(table_out, index=False)

    plot_path = plot_accuracy(result_df, args.out_dir)

    print(f"Saved optimal split CSV: {split_out}")
    print(f"Saved results CSV: {result_out}")
    print(f"Saved table CSV: {table_out}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
