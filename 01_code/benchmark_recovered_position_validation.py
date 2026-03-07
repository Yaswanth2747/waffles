import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from ann_index import retrieve_ivf_candidates
from activation_corruption import (
    drop_packets_preserve_order,
    reconstruct_from_observed_packets,
    sample_missing_positions,
    split_tensor_packets,
)
from corpus_utils import build_loader, infer_dataset_name
from models import SUPPORTED_MODELS, load_model
from single_query_missing_position_search import (
    build_corpus_signatures,
    normalize_packets,
    packet_signature,
    packets_to_matrix,
    preprocess_corpus_packets,
    score_corpus_candidate,
    select_candidate_indices,
)
from splitter import SplitModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark hidden missing-position recovery over many validation queries."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/ and val/")
    parser.add_argument("--dataset-name", default="auto", choices=["auto", "imagenet", "imagenette"])
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--model-name", default="mobilenet_v2", choices=SUPPORTED_MODELS)
    parser.add_argument("--split-idx", type=int, required=True)
    parser.add_argument("--packet-elems", type=int, default=256)
    parser.add_argument("--missing-pct", type=float, default=20.0)
    parser.add_argument("--sample-count", type=int, default=3, help="Random placement samples per query.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--aligner",
        default="greedy_window",
        choices=["exact", "banded_dp", "greedy_window"],
        help="Alignment method used to recover missing positions.",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=50,
        help="Prune corpus search to top-K full-activation cosine matches. Use 0 to search all.",
    )
    parser.add_argument("--ann-index-path", default=None, help="Optional IVF ANN index built from the same corpus.")
    parser.add_argument("--ann-nprobe", type=int, default=4, help="Number of IVF clusters to probe if ANN index is used.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="recovered_position_validation")
    return parser.parse_args()


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


def finalize_metrics(name: str, metrics):
    total = metrics["total"]
    return {
        "mode": name,
        "samples": total,
        "top1_acc": 100.0 * metrics["top1"] / total,
        "top5_acc": 100.0 * metrics["top5"] / total,
        "avg_ce_loss": metrics["loss"] / total,
    }


def reconstruct_act(observed_packets, original_shape, packet_elems, missing_positions, dtype, device):
    return reconstruct_from_observed_packets(
        observed_packets,
        original_shape=original_shape,
        packet_elems=packet_elems,
        missing_positions=missing_positions,
        dtype=dtype,
        device=device,
    )


def plot_summary(df_metrics: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    modes = ["clean", "oracle", "rematched", "random"]
    df_metrics["mode"] = pd.Categorical(df_metrics["mode"], categories=modes, ordered=True)
    df_metrics = df_metrics.sort_values("mode")

    axes[0].bar(df_metrics["mode"], df_metrics["top1_acc"])
    axes[0].set_title("Top-1 Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(df_metrics["mode"], df_metrics["top5_acc"])
    axes[1].set_title("Top-5 Accuracy")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(df_metrics["mode"], df_metrics["avg_ce_loss"])
    axes[2].set_title("Average CE Loss")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "recovered_position_inference_summary.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_recovery(df_queries: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    exact_rate = 100.0 * df_queries["exact_match"].mean()
    overlap_pct = 100.0 * df_queries["overlap_fraction"].mean()
    axes[0].bar(["exact_match_rate", "mean_overlap"], [exact_rate, overlap_pct])
    axes[0].set_ylabel("Percent (%)")
    axes[0].set_title("Recovery Quality")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].hist(df_queries["overlap_fraction"], bins=10, range=(0, 1))
    axes[1].set_title("Overlap Fraction Distribution")
    axes[1].set_xlabel("Recovered overlap fraction")
    axes[1].set_ylabel("Queries")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "recovered_position_recovery_quality.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    corpus = torch.load(args.corpus_path, map_location="cpu")
    if "raw_activations" not in corpus:
        raise ValueError("Corpus file does not contain raw_activations. Rebuild the corpus.")

    corpus_packets_all = preprocess_corpus_packets(corpus["raw_activations"], args.packet_elems)
    corpus_signatures = build_corpus_signatures(corpus_packets_all)
    ann_index = None
    if args.ann_index_path:
        ann_index = torch.load(args.ann_index_path, map_location="cpu")
        if ann_index["corpus_path"] != args.corpus_path:
            raise ValueError("ANN index corpus_path does not match the provided corpus.")

    dataset_name = infer_dataset_name(args.data_dir, args.dataset_name)
    dataset, loader, target_remap = build_loader(
        args.data_dir,
        split="val",
        batch_size=1,
        max_samples=args.max_samples,
        dataset_name=dataset_name,
    )

    model = load_model(args.model_name).to(device).eval()
    split_model = SplitModel(model, args.split_idx)

    clean_metrics = init_metrics()
    oracle_metrics = init_metrics()
    rematched_metrics = init_metrics()
    random_metrics = init_metrics()
    query_rows = []
    query_times = []

    with torch.no_grad():
        for sample_idx, (x, y_local) in enumerate(loader):
            t_query0 = time.perf_counter()
            x = x.to(device)
            y = target_remap[y_local].to(device)
            query_path = dataset.dataset.samples[dataset.indices[sample_idx]][0] if hasattr(dataset, "indices") else dataset.samples[sample_idx][0]

            query_act = split_model.edge_forward(x).squeeze(0).detach().cpu()
            total_packets = len(split_tensor_packets(query_act, args.packet_elems))
            missing_packets = int(round(total_packets * args.missing_pct / 100.0))
            true_missing = sample_missing_positions(total_packets, missing_packets, args.seed + sample_idx)
            observed_packets = drop_packets_preserve_order(query_act, args.packet_elems, true_missing)
            observed_matrix = packets_to_matrix(normalize_packets(observed_packets), args.packet_elems)
            query_signature = packet_signature(observed_matrix)
            if ann_index:
                candidate_indices, _, candidate_pool_size = retrieve_ivf_candidates(
                    query_signature,
                    ann_index["signature_matrix"],
                    {
                        "centroids": ann_index["centroids"],
                        "assignments": ann_index["assignments"],
                        "inverted_lists": ann_index["inverted_lists"],
                    },
                    top_k=args.top_k_candidates,
                    nprobe=args.ann_nprobe,
                )
            else:
                candidate_indices, _ = select_candidate_indices(
                    query_signature,
                    corpus_signatures,
                    args.top_k_candidates,
                )
                candidate_pool_size = len(candidate_indices)

            best_row = None
            for corpus_idx in candidate_indices:
                score, aligned_positions, recovered_missing = score_corpus_candidate(
                    observed_matrix,
                    corpus_packets_all[corpus_idx],
                    aligner=args.aligner,
                )
                overlap = len(set(recovered_missing) & set(true_missing))
                row = {
                    "corpus_idx": corpus_idx,
                    "score": score,
                    "recovered_missing": recovered_missing,
                    "aligned_positions": aligned_positions,
                    "overlap": overlap,
                    "exact_match": recovered_missing == true_missing,
                    "corpus_label_imagenet": int(corpus["labels_imagenet"][corpus_idx].item()),
                }
                if best_row is None or row["score"] > best_row["score"]:
                    best_row = row

            clean_logits = split_model.server_forward(query_act.unsqueeze(0).to(device))
            update_metrics(clean_metrics, clean_logits, y)

            oracle_act = reconstruct_act(
                observed_packets,
                tuple(query_act.shape),
                args.packet_elems,
                true_missing,
                query_act.dtype,
                device,
            )
            oracle_logits = split_model.server_forward(oracle_act.unsqueeze(0))
            update_metrics(oracle_metrics, oracle_logits, y)

            rematched_act = reconstruct_act(
                observed_packets,
                tuple(query_act.shape),
                args.packet_elems,
                best_row["recovered_missing"],
                query_act.dtype,
                device,
            )
            rematched_logits = split_model.server_forward(rematched_act.unsqueeze(0))
            update_metrics(rematched_metrics, rematched_logits, y)

            for hypothesis_idx in range(args.sample_count):
                random_missing = sample_missing_positions(
                    total_packets,
                    missing_packets,
                    args.seed + 100000 + sample_idx * args.sample_count + hypothesis_idx,
                )
                random_act = reconstruct_act(
                    observed_packets,
                    tuple(query_act.shape),
                    args.packet_elems,
                    random_missing,
                    query_act.dtype,
                    device,
                )
                random_logits = split_model.server_forward(random_act.unsqueeze(0))
                update_metrics(random_metrics, random_logits, y)

            query_rows.append(
                {
                    "query_index": sample_idx,
                    "query_path": query_path,
                    "query_label_imagenet": int(y.item()),
                    "total_packets": total_packets,
                    "missing_packets": missing_packets,
                    "true_missing": str(true_missing),
                    "recovered_missing": str(best_row["recovered_missing"]),
                    "best_corpus_idx": best_row["corpus_idx"],
                    "best_corpus_label_imagenet": best_row["corpus_label_imagenet"],
                    "best_score": best_row["score"],
                    "exact_match": best_row["exact_match"],
                    "overlap": best_row["overlap"],
                    "overlap_fraction": best_row["overlap"] / max(1, missing_packets),
                    "candidate_pool_size": candidate_pool_size,
                    "aligner": args.aligner,
                }
            )
            query_times.append(time.perf_counter() - t_query0)

    df_queries = pd.DataFrame(query_rows)
    df_metrics = pd.DataFrame(
        [
            finalize_metrics("clean", clean_metrics),
            finalize_metrics("oracle", oracle_metrics),
            {
                **finalize_metrics("random", random_metrics),
                "samples": random_metrics["total"],
            },
            finalize_metrics("rematched", rematched_metrics),
        ]
    )

    exact_match_rate = 100.0 * df_queries["exact_match"].mean()
    mean_overlap = df_queries["overlap"].mean()
    mean_overlap_fraction = 100.0 * df_queries["overlap_fraction"].mean()
    recovery_summary = pd.DataFrame(
        [
            {
                "queries": len(df_queries),
                "missing_pct": args.missing_pct,
                "aligner": args.aligner,
                "exact_match_rate_pct": exact_match_rate,
                "mean_overlap_packets": mean_overlap,
                "mean_overlap_fraction_pct": mean_overlap_fraction,
                "mean_query_time_sec": sum(query_times) / len(query_times),
            }
        ]
    )

    queries_out = os.path.join(args.out_dir, "recovered_position_query_results.csv")
    metrics_out = os.path.join(args.out_dir, "recovered_position_inference_metrics.csv")
    summary_out = os.path.join(args.out_dir, "recovered_position_recovery_summary.csv")
    df_queries.to_csv(queries_out, index=False)
    df_metrics.to_csv(metrics_out, index=False)
    recovery_summary.to_csv(summary_out, index=False)

    summary_plot = plot_summary(df_metrics, args.out_dir)
    recovery_plot = plot_recovery(df_queries, args.out_dir)

    print(f"queries={len(df_queries)}")
    print(f"aligner={args.aligner}")
    print(f"missing_pct={args.missing_pct}")
    print(f"exact_match_rate_pct={exact_match_rate:.2f}")
    print(f"mean_overlap_packets={mean_overlap:.2f}")
    print(f"mean_overlap_fraction_pct={mean_overlap_fraction:.2f}")
    print(f"mean_query_time_sec={sum(query_times) / len(query_times):.4f}")
    for _, row in df_metrics.iterrows():
        print(
            f"mode={row['mode']} top1={row['top1_acc']:.2f}% "
            f"top5={row['top5_acc']:.2f}% ce={row['avg_ce_loss']:.4f}"
        )
    print(f"saved_queries={queries_out}")
    print(f"saved_metrics={metrics_out}")
    print(f"saved_summary={summary_out}")
    print(f"saved_plot={summary_plot}")
    print(f"saved_plot={recovery_plot}")


if __name__ == "__main__":
    main()
