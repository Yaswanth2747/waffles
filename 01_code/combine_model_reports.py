import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine ViT and MobileNet bandwidth validation reports into overlay CSV/plots."
    )
    parser.add_argument(
        "--vit-dir",
        default="bandwidth_split_validation_report_vit_gpu",
        help="Directory containing the ViT report outputs.",
    )
    parser.add_argument(
        "--mobilenet-dir",
        default="bandwidth_split_validation_report_mobilenet_gpu",
        help="Directory containing the MobileNet report outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="combined_model_comparison",
        help="Directory to write the combined CSV and plots.",
    )
    return parser.parse_args()


def load_model_report(report_dir: str, model_name: str):
    latency_path = os.path.join(report_dir, "optimal_splits_by_bandwidth.csv")
    validation_path = os.path.join(report_dir, "validation_report_by_bandwidth.csv")

    latency_df = pd.read_csv(latency_path).copy()
    latency_df["model_name"] = model_name

    validation_df = pd.read_csv(validation_path).copy()
    validation_df["model_name"] = model_name
    return latency_df, validation_df


def write_combined_csv(latency_df: pd.DataFrame, validation_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    latency_out = os.path.join(out_dir, "combined_latency_by_bandwidth.csv")
    validation_out = os.path.join(out_dir, "combined_validation_by_bandwidth.csv")
    summary_out = os.path.join(out_dir, "combined_summary.csv")

    latency_df.to_csv(latency_out, index=False)
    validation_df.to_csv(validation_out, index=False)

    summary_df = validation_df.merge(
        latency_df[
            [
                "model_name",
                "bandwidth_mbps",
                "optimal_split_idx",
                "edge_ms",
                "server_ms",
                "comm_ms",
                "total_ms",
                "activation_mb",
            ]
        ],
        on=["model_name", "bandwidth_mbps"],
        how="left",
    )
    summary_df.to_csv(summary_out, index=False)

    return latency_out, validation_out, summary_out


def plot_latency(latency_df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, model_df in latency_df.groupby("model_name"):
        model_df = model_df.sort_values("bandwidth_mbps")
        ax.plot(
            model_df["bandwidth_mbps"],
            model_df["total_ms"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_title("Total Latency vs Bandwidth")
    ax.set_xlabel("Bandwidth (MB/s)")
    ax.set_ylabel("Total Latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "combined_total_latency.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_accuracy(validation_df: pd.DataFrame, metric_col: str, title: str, out_name: str, out_dir: str):
    missing_levels = sorted(validation_df["missing_packets"].unique())
    fig, axes = plt.subplots(1, len(missing_levels), figsize=(6 * len(missing_levels), 5), sharey=True)
    if len(missing_levels) == 1:
        axes = [axes]

    for ax, missing_packets in zip(axes, missing_levels):
        subset = validation_df[validation_df["missing_packets"] == missing_packets]
        for (model_name, mode), group in subset.groupby(["model_name", "mode"]):
            group = group.sort_values("bandwidth_mbps")
            ax.plot(
                group["bandwidth_mbps"],
                group[metric_col],
                marker="o",
                linewidth=2,
                label=f"{model_name}:{mode}",
            )
        ax.set_title(f"k = {missing_packets}")
        ax.set_xlabel("Bandwidth (MB/s)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_publication_figure(latency_df: pd.DataFrame, validation_df: pd.DataFrame, out_dir: str):
    missing_levels = sorted(validation_df["missing_packets"].unique())
    fig, axes = plt.subplots(
        3,
        len(missing_levels),
        figsize=(6 * len(missing_levels), 13),
        sharex="row",
    )

    if len(missing_levels) == 1:
        axes = axes.reshape(3, 1)

    latency_ax = axes[0, 0]
    for model_name, model_df in latency_df.groupby("model_name"):
        model_df = model_df.sort_values("bandwidth_mbps")
        latency_ax.plot(
            model_df["bandwidth_mbps"],
            model_df["total_ms"],
            marker="o",
            linewidth=2.5,
            label=model_name,
        )
    latency_ax.set_title("Total Latency vs Bandwidth")
    latency_ax.set_xlabel("Bandwidth (MB/s)")
    latency_ax.set_ylabel("Total Latency (ms)")
    latency_ax.grid(True, alpha=0.3)
    latency_ax.legend()

    for col in range(1, len(missing_levels)):
        axes[0, col].axis("off")

    row_specs = [
        ("top1_acc", "Top-1 Accuracy (%)"),
        ("top5_acc", "Top-5 Accuracy (%)"),
    ]
    mode_styles = {
        "clean": "-",
        "oracle": "--",
        "random": ":",
    }

    for row_idx, (metric_col, ylabel) in enumerate(row_specs, start=1):
        for col_idx, missing_packets in enumerate(missing_levels):
            ax = axes[row_idx, col_idx]
            subset = validation_df[validation_df["missing_packets"] == missing_packets]
            for (model_name, mode), group in subset.groupby(["model_name", "mode"]):
                group = group.sort_values("bandwidth_mbps")
                ax.plot(
                    group["bandwidth_mbps"],
                    group[metric_col],
                    marker="o",
                    linewidth=2,
                    linestyle=mode_styles.get(mode, "-"),
                    label=f"{model_name}:{mode}",
                )
            ax.set_title(f"k = {missing_packets}")
            ax.set_xlabel("Bandwidth (MB/s)")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.grid(True, alpha=0.3)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, "combined_publication_figure.png")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()

    vit_latency, vit_validation = load_model_report(args.vit_dir, "vit_b_16")
    mb_latency, mb_validation = load_model_report(args.mobilenet_dir, "mobilenet_v2")

    latency_df = pd.concat([vit_latency, mb_latency], ignore_index=True)
    validation_df = pd.concat([vit_validation, mb_validation], ignore_index=True)

    latency_out, validation_out, summary_out = write_combined_csv(latency_df, validation_df, args.out_dir)
    latency_plot = plot_latency(latency_df, args.out_dir)
    top1_plot = plot_accuracy(
        validation_df,
        metric_col="top1_acc",
        title="Top-1 Accuracy (%)",
        out_name="combined_top1_overlay.png",
        out_dir=args.out_dir,
    )
    top5_plot = plot_accuracy(
        validation_df,
        metric_col="top5_acc",
        title="Top-5 Accuracy (%)",
        out_name="combined_top5_overlay.png",
        out_dir=args.out_dir,
    )
    publication_plot = plot_publication_figure(latency_df, validation_df, args.out_dir)

    print(f"Saved combined latency CSV: {latency_out}")
    print(f"Saved combined validation CSV: {validation_out}")
    print(f"Saved combined summary CSV: {summary_out}")
    print(f"Saved latency plot: {latency_plot}")
    print(f"Saved top-1 plot: {top1_plot}")
    print(f"Saved top-5 plot: {top5_plot}")
    print(f"Saved publication figure: {publication_plot}")


if __name__ == "__main__":
    main()
