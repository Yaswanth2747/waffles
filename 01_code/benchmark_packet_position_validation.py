import argparse
import os
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
from models import SUPPORTED_MODELS, load_model
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
        description="Benchmark clean/oracle/random packet-position inference on a validation dataset"
    )
    parser.add_argument("--val-dir", required=True, help="ImageFolder validation directory")
    parser.add_argument("--dataset-name", default="auto", choices=["auto", "imagenet", "imagenette"])
    parser.add_argument("--model-name", default="vit_b_16", choices=SUPPORTED_MODELS)
    parser.add_argument("--split-idx", type=int, default=8)
    parser.add_argument("--packet-elems", type=int, default=256)
    parser.add_argument("--missing-packets", type=int, default=20)
    parser.add_argument(
        "--missing-packets-levels",
        default=None,
        help="Comma-separated packet-loss levels to evaluate. Overrides --missing-packets.",
    )
    parser.add_argument("--sample-count", type=int, default=5, help="Random hypotheses per image")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", default="packet_position_validation_outputs")
    return parser.parse_args()


def parse_missing_levels(args) -> List[int]:
    if args.missing_packets_levels:
        return [int(x.strip()) for x in args.missing_packets_levels.split(",") if x.strip()]
    return [args.missing_packets]


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return "cpu"
    return device_arg


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


def init_metrics():
    return {"total": 0, "loss": 0.0, "top1": 0, "top5": 0}


def update_metrics(metrics, logits: torch.Tensor, target: torch.Tensor):
    loss = F.cross_entropy(logits, target)
    batch = target.shape[0]
    metrics["total"] += batch
    metrics["loss"] += float(loss.item()) * batch
    metrics["top1"] += topk_correct(logits, target, 1)
    metrics["top5"] += topk_correct(logits, target, 5)


def finalize_metrics(metrics, label: str, total_packets: int, missing_packets: int, sample_count: int):
    total = metrics["total"]
    return {
        "mode": label,
        "total_packets": total_packets,
        "missing_packets": missing_packets,
        "sample_count": sample_count,
        "top1_acc": 100.0 * metrics["top1"] / total,
        "top5_acc": 100.0 * metrics["top5"] / total,
        "avg_ce_loss": metrics["loss"] / total,
        "samples": total,
    }


def save_plot(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    mode_colors = {"clean": "green", "oracle": "crimson", "random": "tab:blue"}

    for mode in ["clean", "oracle", "random"]:
        df_mode = df[df["mode"] == mode].sort_values("missing_packets")
        ax[0].plot(
            df_mode["missing_packets"],
            df_mode["top1_acc"],
            marker="o",
            label=mode,
            color=mode_colors[mode],
        )
        ax[1].plot(
            df_mode["missing_packets"],
            df_mode["top5_acc"],
            marker="o",
            label=mode,
            color=mode_colors[mode],
        )
        ax[2].plot(
            df_mode["missing_packets"],
            df_mode["avg_ce_loss"],
            marker="o",
            label=mode,
            color=mode_colors[mode],
        )

    ax[0].set_title("Top-1 Accuracy vs Missing Packets")
    ax[0].set_ylabel("%")
    ax[1].set_title("Top-5 Accuracy vs Missing Packets")
    ax[1].set_ylabel("%")
    ax[2].set_title("Average CE Loss vs Missing Packets")
    ax[2].set_ylabel("Loss")

    for a in ax:
        a.set_xlabel("Missing Packets")
        a.grid(True, alpha=0.3)
        a.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "packet_position_validation.png"), dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    dataset_name = infer_dataset_name(args.val_dir, args.dataset_name)
    loader, target_remap = build_loader(args.val_dir, args.max_samples, dataset_name)

    model = load_model(model_name=args.model_name, device=device)
    split = SplitModel(model, args.split_idx)
    missing_levels = parse_missing_levels(args)

    rows = []

    for missing_level in missing_levels:
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
                    total_packets = get_total_packets(clean_act, args.packet_elems)
                missing_packets = min(missing_level, total_packets)
                oracle_missing = sample_missing_positions(total_packets, missing_packets, args.seed + sample_idx)
                observed_packets = drop_packets_preserve_order(clean_act.cpu(), args.packet_elems, oracle_missing)

                clean_logits = split.server_forward(clean_act)
                update_metrics(clean_metrics, clean_logits, y)

                oracle_act = reconstruct_from_observed_packets(
                    observed_packets,
                    original_shape=tuple(clean_act.shape),
                    packet_elems=args.packet_elems,
                    missing_positions=oracle_missing,
                    dtype=clean_act.dtype,
                    device=device,
                )
                oracle_logits = split.server_forward(oracle_act)
                update_metrics(oracle_metrics, oracle_logits, y)

                for hypothesis_idx in range(args.sample_count):
                    candidate_missing = sample_missing_positions(
                        total_packets,
                        missing_packets,
                        args.seed + 100000 + sample_idx * args.sample_count + hypothesis_idx,
                    )
                    candidate_act = reconstruct_from_observed_packets(
                        observed_packets,
                        original_shape=tuple(clean_act.shape),
                        packet_elems=args.packet_elems,
                        missing_positions=candidate_missing,
                        dtype=clean_act.dtype,
                        device=device,
                    )
                    candidate_logits = split.server_forward(candidate_act)
                    update_metrics(random_metrics, candidate_logits, y)

        rows.extend(
            [
                finalize_metrics(clean_metrics, "clean", total_packets, missing_packets, 1),
                finalize_metrics(oracle_metrics, "oracle", total_packets, missing_packets, 1),
                finalize_metrics(random_metrics, "random", total_packets, missing_packets, args.sample_count),
            ]
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "packet_position_validation.csv"), index=False)
    save_plot(df, args.out_dir)

    for row in rows:
        print(
            f"mode={row['mode']} total_packets={row['total_packets']} "
            f"missing_packets={row['missing_packets']} sample_count={row['sample_count']} "
            f"top1={row['top1_acc']:.2f}% top5={row['top5_acc']:.2f}% "
            f"ce_loss={row['avg_ce_loss']:.4f}"
        )
    print(f"Model used: {args.model_name}")
    print(f"Dataset mapping used: {dataset_name}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
