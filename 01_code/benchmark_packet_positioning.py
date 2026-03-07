import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from activation_corruption import (
    drop_packets_preserve_order,
    get_total_packets,
    reconstruct_from_observed_packets,
    sample_missing_positions,
)
from models import SUPPORTED_MODELS, load_model
from splitter import SplitModel


DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark unknown packet-position hypotheses against oracle and clean inference"
    )
    parser.add_argument("--model-name", type=str, default="vit_b_16", choices=SUPPORTED_MODELS, help="Model to run")
    parser.add_argument("--split-idx", type=int, default=8, help="ViT split index")
    parser.add_argument("--image-path", type=str, default=None, help="Local image path")
    parser.add_argument("--image-url", type=str, default=DEFAULT_IMAGE_URL, help="Input image URL")
    parser.add_argument("--packet-elems", type=int, default=256, help="Packet size in float elements")
    parser.add_argument("--missing-packets", type=int, default=20, help="Number of missing packets k")
    parser.add_argument("--sample-count", type=int, default=100, help="Number of random hypotheses S")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--target-class-idx", type=int, default=None, help="Optional class index for CE loss")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device")
    parser.add_argument("--out-dir", default="packet_position_benchmark_outputs", help="Output directory")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return "cpu"
    return device_arg


def load_input_image(args):
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
    else:
        response = requests.get(args.image_url, stream=True, timeout=20)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

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
    return transform(image).unsqueeze(0), image


def load_labels():
    try:
        response = requests.get(LABELS_URL, timeout=20)
        response.raise_for_status()
        labels = [line for line in response.text.splitlines() if line]
        if len(labels) >= 1000:
            return labels
    except Exception:
        pass
    return [f"class_{i}" for i in range(1000)]


def summarize_logits(logits: torch.Tensor, labels, reference_class_idx: int, target_class_idx):
    probs = F.softmax(logits, dim=1)
    top_prob, top_idx = torch.max(probs, dim=1)
    summary = {
        "pred_idx": int(top_idx.item()),
        "pred_label": labels[int(top_idx.item())] if int(top_idx.item()) < len(labels) else f"class_{int(top_idx.item())}",
        "pred_prob": float(top_prob.item()),
        "ref_prob": float(probs[0, reference_class_idx].item()),
    }
    if target_class_idx is not None:
        target = torch.tensor([target_class_idx], dtype=torch.long, device=logits.device)
        summary["ce_loss"] = float(F.cross_entropy(logits, target).item())
    return summary


def run_case(split, activation: torch.Tensor):
    with torch.no_grad():
        return split.server_forward(activation)


def plot_results(df_random: pd.DataFrame, oracle_row: dict, clean_row: dict, out_dir: str):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].hist(df_random["ref_prob"], bins=20, color="tab:blue", alpha=0.8)
    ax[0].axvline(clean_row["ref_prob"], color="green", linewidth=2, label="clean")
    ax[0].axvline(oracle_row["ref_prob"], color="crimson", linewidth=2, label="oracle")
    ax[0].set_title("Reference-Class Probability")
    ax[0].set_xlabel("Probability")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].hist(df_random["logit_l2"], bins=20, color="tab:orange", alpha=0.8)
    ax[1].axvline(oracle_row["logit_l2"], color="crimson", linewidth=2, label="oracle")
    ax[1].set_title("Logit L2 Distance to Clean")
    ax[1].set_xlabel("L2 distance")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    agreement_pct = 100.0 * df_random["matches_clean_top1"].mean()
    ax[2].bar(["random", "oracle", "clean"], [agreement_pct, 100.0 * oracle_row["matches_clean_top1"], 100.0], color=["tab:blue", "crimson", "green"])
    ax[2].set_title("Top-1 Agreement With Clean")
    ax[2].set_ylabel("Agreement (%)")
    ax[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "packet_positioning_comparison.png"), dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    labels = load_labels()
    x, _ = load_input_image(args)
    x = x.to(device)

    model = load_model(model_name=args.model_name, device=device)
    split = SplitModel(model, args.split_idx)

    with torch.no_grad():
        clean_act = split.edge_forward(x)
        clean_logits = split.server_forward(clean_act)

    total_packets = get_total_packets(clean_act, args.packet_elems)
    missing_packets = min(args.missing_packets, total_packets)
    total_combinations = math.comb(total_packets, missing_packets)
    oracle_missing = sample_missing_positions(total_packets, missing_packets, args.seed)
    observed_packets = drop_packets_preserve_order(clean_act.cpu(), args.packet_elems, oracle_missing)

    clean_top1 = int(clean_logits.argmax(dim=1).item())
    clean_summary = summarize_logits(clean_logits, labels, clean_top1, args.target_class_idx)
    clean_summary["case"] = "clean"
    clean_summary["logit_l2"] = 0.0
    clean_summary["matches_clean_top1"] = True

    oracle_act = reconstruct_from_observed_packets(
        observed_packets,
        original_shape=tuple(clean_act.shape),
        packet_elems=args.packet_elems,
        missing_positions=oracle_missing,
        dtype=clean_act.dtype,
        device=device,
    )
    oracle_logits = run_case(split, oracle_act)
    oracle_summary = summarize_logits(oracle_logits, labels, clean_top1, args.target_class_idx)
    oracle_summary["case"] = "oracle"
    oracle_summary["logit_l2"] = float(torch.norm(oracle_logits - clean_logits).item())
    oracle_summary["matches_clean_top1"] = oracle_summary["pred_idx"] == clean_top1

    rows = []
    for sample_idx in range(args.sample_count):
        candidate_missing = sample_missing_positions(
            total_packets,
            missing_packets,
            args.seed + 1000 + sample_idx,
        )
        candidate_act = reconstruct_from_observed_packets(
            observed_packets,
            original_shape=tuple(clean_act.shape),
            packet_elems=args.packet_elems,
            missing_positions=candidate_missing,
            dtype=clean_act.dtype,
            device=device,
        )
        candidate_logits = run_case(split, candidate_act)
        row = summarize_logits(candidate_logits, labels, clean_top1, args.target_class_idx)
        row["case"] = "random"
        row["sample_idx"] = sample_idx
        row["logit_l2"] = float(torch.norm(candidate_logits - clean_logits).item())
        row["matches_clean_top1"] = row["pred_idx"] == clean_top1
        row["matches_oracle_positions"] = candidate_missing == oracle_missing
        rows.append(row)

    df_random = pd.DataFrame(rows)
    df_random.to_csv(os.path.join(args.out_dir, "packet_positioning_random_samples.csv"), index=False)
    plot_results(df_random, oracle_summary, clean_summary, args.out_dir)

    print(f"total_packets={total_packets}")
    print(f"missing_packets={missing_packets}")
    print(f"total_combinations=N_choose_k={total_combinations}")
    print(f"sample_count={args.sample_count}")
    print(f"model_name={args.model_name}")
    print(f"oracle_missing_positions={oracle_missing}")
    print(
        f"clean: pred={clean_summary['pred_label']} prob={clean_summary['pred_prob']*100:.2f}%"
    )
    print(
        f"oracle: pred={oracle_summary['pred_label']} prob={oracle_summary['pred_prob']*100:.2f}% "
        f"logit_l2={oracle_summary['logit_l2']:.4f}"
    )
    if args.target_class_idx is not None:
        print(
            f"clean_ce_loss={clean_summary['ce_loss']:.6f} "
            f"oracle_ce_loss={oracle_summary['ce_loss']:.6f} "
            f"random_mean_ce_loss={df_random['ce_loss'].mean():.6f}"
        )
    print(
        f"random_top1_match_with_clean={100.0 * df_random['matches_clean_top1'].mean():.2f}%"
    )
    print(
        f"random_ref_prob_mean={df_random['ref_prob'].mean()*100:.2f}% "
        f"random_ref_prob_std={df_random['ref_prob'].std():.2f}"
    )
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
