import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from activation_corruption import drop_packets_preserve_order, reconstruct_from_observed_packets, sample_missing_positions
from corpus_utils import build_loader, infer_dataset_name, packet_rank_summary, packet_similarity_matrix
from models import SUPPORTED_MODELS, load_model
from splitter import SplitModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-query packet-ranking sanity check against an activation corpus."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/ and val/")
    parser.add_argument("--dataset-name", default="auto", choices=["auto", "imagenet", "imagenette"])
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--model-name", default="mobilenet_v2", choices=SUPPORTED_MODELS)
    parser.add_argument("--split-idx", type=int, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--packet-elems", type=int, default=256)
    parser.add_argument("--missing-pct", type=float, default=20.0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", default="single_query_packet_ranking")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def plot_similarity(sim: torch.Tensor, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim.numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Corpus packet position")
    ax.set_ylabel("Query packet index")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    corpus = torch.load(args.corpus_path, map_location="cpu")
    dataset_name = infer_dataset_name(args.data_dir, args.dataset_name)
    dataset, loader, target_remap = build_loader(
        args.data_dir,
        split="val",
        batch_size=1,
        max_samples=0,
        dataset_name=dataset_name,
    )

    model = load_model(args.model_name).to(device).eval()
    split_model = SplitModel(model, args.split_idx)

    sample = dataset[args.query_index]
    x, y_local = sample
    x = x.unsqueeze(0).to(device)
    y_imagenet = int(target_remap[torch.tensor([y_local])][0].item())
    query_path = dataset.samples[args.query_index][0]

    with torch.no_grad():
        query_act = split_model.edge_forward(x).squeeze(0).detach().cpu()
        query_flat = F.normalize(query_act.reshape(1, -1).float(), p=2, dim=1)
        sims = torch.matmul(query_flat, corpus["activations"].T).squeeze(0)
        best_idx = int(torch.argmax(sims).item())
        best_score = float(sims[best_idx].item())

    matched_act = corpus["activations"][best_idx]
    if "raw_activations" not in corpus:
        raise ValueError("Corpus file does not contain raw_activations. Rebuild the corpus with the current build_activation_corpus.py.")
    matched_raw = corpus["raw_activations"][best_idx]

    sim_clean = packet_similarity_matrix(query_act, matched_raw, args.packet_elems)
    all_positions = list(range(sim_clean.shape[0]))
    clean_top1, clean_top3, clean_top5, clean_mean_rank = packet_rank_summary(sim_clean, all_positions)

    total_packets = sim_clean.shape[0]
    missing_packets = int(round(total_packets * args.missing_pct / 100.0))
    random.seed(args.seed)
    missing_positions = sample_missing_positions(total_packets, missing_packets, args.seed)
    observed_packets = drop_packets_preserve_order(query_act, args.packet_elems, missing_positions)
    observed_query = reconstruct_from_observed_packets(
        observed_packets,
        original_shape=tuple(query_act.shape),
        packet_elems=args.packet_elems,
        missing_positions=missing_positions,
        dtype=query_act.dtype,
        device="cpu",
    )
    sim_observed = packet_similarity_matrix(observed_query, matched_raw, args.packet_elems)
    observed_positions = [i for i in range(total_packets) if i not in set(missing_positions)]
    observed_top1, observed_top3, observed_top5, observed_mean_rank = packet_rank_summary(
        sim_observed[observed_positions],
        observed_positions,
    )

    clean_plot = os.path.join(args.out_dir, "packet_similarity_clean.png")
    observed_plot = os.path.join(args.out_dir, "packet_similarity_observed.png")
    plot_similarity(sim_clean, clean_plot, "Clean Query vs Matched Corpus Activation")
    plot_similarity(sim_observed, observed_plot, "Observed Query vs Matched Corpus Activation")

    print(f"query_path={query_path}")
    print(f"query_label_imagenet={y_imagenet}")
    print(f"matched_corpus_index={best_idx}")
    print(f"matched_corpus_path={corpus['paths'][best_idx]}")
    print(f"matched_label_imagenet={int(corpus['labels_imagenet'][best_idx].item())}")
    print(f"full_activation_cosine={best_score:.6f}")
    print(f"total_packets={total_packets}")
    print(f"missing_pct={args.missing_pct}")
    print(f"missing_packets={missing_packets}")
    print(
        f"clean_packet_rank top1={clean_top1:.2f}% top3={clean_top3:.2f}% "
        f"top5={clean_top5:.2f}% mean_rank={clean_mean_rank:.2f}"
    )
    print(
        f"observed_packet_rank top1={observed_top1:.2f}% top3={observed_top3:.2f}% "
        f"top5={observed_top5:.2f}% mean_rank={observed_mean_rank:.2f}"
    )
    print(f"saved_plot={clean_plot}")
    print(f"saved_plot={observed_plot}")


if __name__ == "__main__":
    main()
