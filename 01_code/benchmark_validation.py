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

from activation_corruption import get_total_packets, zero_activation_packets
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
        description="Benchmark split inference degradation on a validation dataset"
    )
    parser.add_argument("--val-dir", required=True, help="ImageFolder validation directory")
    parser.add_argument("--model-name", default="vit_b_16", choices=SUPPORTED_MODELS)
    parser.add_argument("--split-idx", type=int, default=8, help="ViT split index")
    parser.add_argument(
        "--loss-levels",
        default="0,1,5,10,20",
        help="Comma-separated packet zero percentages",
    )
    parser.add_argument(
        "--packet-elems",
        type=int,
        default=256,
        help="Packet size in float elements for corruption",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of validation images to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for corruption",
    )
    parser.add_argument(
        "--out-dir",
        default="validation_benchmark_outputs",
        help="Output directory for csv and plots",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Benchmark device",
    )
    parser.add_argument(
        "--dataset-name",
        default="auto",
        choices=["auto", "imagenet", "imagenette"],
        help="Dataset label mapping to use",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return "cpu"
    return device_arg


def parse_loss_levels(loss_levels: str) -> List[float]:
    return [float(x.strip()) for x in loss_levels.split(",") if x.strip()]


def infer_dataset_name(val_dir: str, dataset_name: str) -> str:
    if dataset_name != "auto":
        return dataset_name
    lowered = val_dir.lower()
    if "imagenette" in lowered:
        return "imagenette"
    return "imagenet"


def resolve_dataset_root(val_dir: str) -> str:
    if os.path.isdir(os.path.join(val_dir, "val")):
        return os.path.join(val_dir, "val")
    return val_dir


def build_target_remap(classes: List[str], dataset_name: str) -> torch.Tensor:
    if dataset_name == "imagenet":
        return torch.tensor(list(range(len(classes))), dtype=torch.long)
    if dataset_name == "imagenette":
        missing = [cls for cls in classes if cls not in IMAGENETTE_WNID_TO_IDX]
        if missing:
            raise ValueError(
                f"Unsupported Imagenette class folders: {missing}. "
                "Expected WNID folder names."
            )
        return torch.tensor(
            [IMAGENETTE_WNID_TO_IDX[cls] for cls in classes],
            dtype=torch.long,
        )
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


def run_level(
    split,
    loader,
    target_remap: torch.Tensor,
    level_pct: float,
    packet_elems: int,
    device: str,
    seed: int,
):
    total = 0
    total_loss = 0.0
    top1 = 0
    top5 = 0
    first_packet_count = None
    first_zero_packets = None

    with torch.no_grad():
        for sample_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = target_remap[y].to(device)

            act = split.edge_forward(x)
            packet_count = get_total_packets(act, packet_elems)
            zero_packets = round(packet_count * level_pct / 100.0)
            act_corrupt, selected = zero_activation_packets(
                act,
                packet_elems=packet_elems,
                zero_packets=zero_packets,
                seed=seed + sample_idx,
            )

            logits = split.server_forward(act_corrupt)
            loss = F.cross_entropy(logits, y)

            total += x.shape[0]
            total_loss += float(loss.item()) * x.shape[0]
            top1 += topk_correct(logits, y, 1)
            top5 += topk_correct(logits, y, 5)

            if first_packet_count is None:
                first_packet_count = packet_count
                first_zero_packets = len(selected)

    return {
        "loss_pct": level_pct,
        "zero_packets": first_zero_packets,
        "total_packets": first_packet_count,
        "top1_acc": 100.0 * top1 / total,
        "top5_acc": 100.0 * top5 / total,
        "avg_ce_loss": total_loss / total,
        "samples": total,
    }


def save_plots(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].plot(df["loss_pct"], df["top1_acc"], marker="o", linewidth=2)
    ax[0].set_title("Top-1 Accuracy vs Packet Zero %")
    ax[0].set_xlabel("Packet Zero %")
    ax[0].set_ylabel("Top-1 Accuracy (%)")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df["loss_pct"], df["top5_acc"], marker="o", linewidth=2)
    ax[1].set_title("Top-5 Accuracy vs Packet Zero %")
    ax[1].set_xlabel("Packet Zero %")
    ax[1].set_ylabel("Top-5 Accuracy (%)")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(df["loss_pct"], df["avg_ce_loss"], marker="o", linewidth=2, color="crimson")
    ax[2].set_title("Cross-Entropy Loss vs Packet Zero %")
    ax[2].set_xlabel("Packet Zero %")
    ax[2].set_ylabel("Average CE Loss")
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "validation_benchmark.png"), dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    dataset_name = infer_dataset_name(args.val_dir, args.dataset_name)
    model = load_model(model_name=args.model_name, device=device)
    split = SplitModel(model, args.split_idx)
    loader, target_remap = build_loader(args.val_dir, args.max_samples, dataset_name)
    levels = parse_loss_levels(args.loss_levels)

    results = []
    for level_pct in levels:
        result = run_level(
            split,
            loader,
            target_remap,
            level_pct=level_pct,
            packet_elems=args.packet_elems,
            device=device,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"loss_pct={level_pct:.2f} total_packets={result['total_packets']} "
            f"zero_packets={result['zero_packets']} top1={result['top1_acc']:.2f}% "
            f"top5={result['top5_acc']:.2f}% ce_loss={result['avg_ce_loss']:.4f}"
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, "validation_benchmark.csv")
    df.to_csv(csv_path, index=False)
    save_plots(df, args.out_dir)
    print(f"Model used: {args.model_name}")
    print(f"Dataset mapping used: {dataset_name}")
    print(f"Saved benchmark outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
