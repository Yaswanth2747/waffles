import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from activation_corruption import drop_packets_preserve_order, sample_missing_positions, split_tensor_packets
from corpus_utils import build_loader, infer_dataset_name
from models import SUPPORTED_MODELS, load_model
from splitter import SplitModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recover hidden missing packet positions for a single query using a training activation corpus."
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-report", type=int, default=5)
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=50,
        help="Prune corpus search to top-K full-activation cosine matches. Use 0 to search all.",
    )
    parser.add_argument("--out-dir", default="single_query_missing_position_search")
    return parser.parse_args()


def normalize_packets(packets):
    out = []
    for pkt in packets:
        pkt = pkt.float()
        norm = torch.linalg.vector_norm(pkt)
        out.append(pkt / norm if norm > 0 else pkt)
    return out


def packets_to_matrix(packets, packet_elems: int):
    mat = torch.zeros(len(packets), packet_elems, dtype=torch.float32)
    for i, pkt in enumerate(packets):
        mat[i, : pkt.numel()] = pkt
    return mat


def preprocess_corpus_packets(raw_activations, packet_elems: int):
    return [
        packets_to_matrix(normalize_packets(split_tensor_packets(act, packet_elems)), packet_elems)
        for act in raw_activations
    ]


def packet_signature(packet_matrix: torch.Tensor):
    sig = packet_matrix.mean(dim=0)
    norm = torch.linalg.vector_norm(sig)
    return sig / norm if norm > 0 else sig


def packet_similarity_matrix(observed_matrix: torch.Tensor, corpus_matrix: torch.Tensor):
    return observed_matrix @ corpus_matrix.T


def best_monotonic_alignment(sim: torch.Tensor):
    obs_count, total_packets = sim.shape
    neg_inf = -1e30
    dp = torch.full((obs_count + 1, total_packets + 1), neg_inf, dtype=torch.float32)
    take = torch.zeros((obs_count + 1, total_packets + 1), dtype=torch.bool)
    dp[0, :] = 0.0

    for i in range(1, obs_count + 1):
        for j in range(1, total_packets + 1):
            skip_score = dp[i, j - 1]
            take_score = dp[i - 1, j - 1] + sim[i - 1, j - 1]
            if take_score > skip_score:
                dp[i, j] = take_score
                take[i, j] = True
            else:
                dp[i, j] = skip_score

    positions = []
    i = obs_count
    j = total_packets
    while i > 0 and j > 0:
        if take[i, j]:
            positions.append(j - 1)
            i -= 1
            j -= 1
        else:
            j -= 1
    positions.reverse()
    missing = [idx for idx in range(total_packets) if idx not in set(positions)]
    return float(dp[obs_count, total_packets].item()), positions, missing


def banded_dp_alignment(sim: torch.Tensor):
    obs_count, total_packets = sim.shape
    missing_packets = total_packets - obs_count
    neg_inf = -1e30
    dp = torch.full((obs_count, missing_packets + 1), neg_inf, dtype=torch.float32)
    prev = torch.full((obs_count, missing_packets + 1), -1, dtype=torch.long)

    for shift in range(missing_packets + 1):
        pos = shift
        dp[0, shift] = sim[0, pos]

    for i in range(1, obs_count):
        for shift in range(missing_packets + 1):
            pos = i + shift
            if pos >= total_packets:
                continue
            best_prev_shift = -1
            best_prev_score = neg_inf
            for prev_shift in range(shift + 1):
                score = dp[i - 1, prev_shift]
                if score > best_prev_score:
                    best_prev_score = score
                    best_prev_shift = prev_shift
            dp[i, shift] = best_prev_score + sim[i, pos]
            prev[i, shift] = best_prev_shift

    last_shift = int(torch.argmax(dp[obs_count - 1]).item())
    best_score = float(dp[obs_count - 1, last_shift].item())

    shifts = [0] * obs_count
    shifts[-1] = last_shift
    for i in range(obs_count - 1, 0, -1):
        shifts[i - 1] = int(prev[i, shifts[i]].item())

    positions = [i + shifts[i] for i in range(obs_count)]
    missing = [idx for idx in range(total_packets) if idx not in set(positions)]
    return best_score, positions, missing


def greedy_window_alignment(sim: torch.Tensor):
    obs_count, total_packets = sim.shape
    missing_packets = total_packets - obs_count
    positions = []
    score = 0.0
    prev_pos = -1

    for i in range(obs_count):
        min_pos = max(prev_pos + 1, i)
        max_pos = min(i + missing_packets, total_packets - (obs_count - i))
        if min_pos > max_pos:
            raise ValueError("Invalid greedy window bounds")

        local = sim[i, min_pos : max_pos + 1]
        offset = int(torch.argmax(local).item())
        pos = min_pos + offset
        positions.append(pos)
        score += float(sim[i, pos].item())
        prev_pos = pos

        if pos == max_pos and pos == total_packets - (obs_count - i):
            for j in range(i + 1, obs_count):
                next_pos = pos + (j - i)
                positions.append(next_pos)
                score += float(sim[j, next_pos].item())
            break

    positions = positions[:obs_count]
    missing = [idx for idx in range(total_packets) if idx not in set(positions)]
    return score, positions, missing


def score_corpus_candidate(observed_matrix: torch.Tensor, corpus_matrix: torch.Tensor, aligner: str = "exact"):
    sim = packet_similarity_matrix(observed_matrix, corpus_matrix)
    if aligner == "exact":
        best_score, aligned_positions, missing_positions = best_monotonic_alignment(sim)
    elif aligner == "banded_dp":
        best_score, aligned_positions, missing_positions = banded_dp_alignment(sim)
    elif aligner == "greedy_window":
        best_score, aligned_positions, missing_positions = greedy_window_alignment(sim)
    else:
        raise ValueError(f"Unsupported aligner: {aligner}")
    return best_score, aligned_positions, missing_positions


def build_corpus_signatures(corpus_packet_mats):
    return torch.stack([packet_signature(mat) for mat in corpus_packet_mats], dim=0)


def select_candidate_indices(query_signature: torch.Tensor, corpus_signatures: torch.Tensor, top_k: int):
    sims = torch.matmul(corpus_signatures, query_signature)
    if top_k and top_k > 0 and top_k < sims.numel():
        return torch.topk(sims, k=top_k).indices.tolist(), sims
    return list(range(sims.numel())), sims


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    corpus = torch.load(args.corpus_path, map_location="cpu")
    if "raw_activations" not in corpus:
        raise ValueError("Corpus file does not contain raw_activations. Rebuild the corpus with the current build_activation_corpus.py.")

    dataset_name = infer_dataset_name(args.data_dir, args.dataset_name)
    dataset, _, target_remap = build_loader(
        args.data_dir,
        split="val",
        batch_size=1,
        max_samples=0,
        dataset_name=dataset_name,
    )

    model = load_model(args.model_name).to(device).eval()
    split_model = SplitModel(model, args.split_idx)

    x, y_local = dataset[args.query_index]
    query_path = dataset.samples[args.query_index][0]
    y_imagenet = int(target_remap[torch.tensor([y_local])][0].item())

    t_edge0 = time.perf_counter()
    with torch.no_grad():
        query_act = split_model.edge_forward(x.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
    t_edge1 = time.perf_counter()

    query_packets = split_tensor_packets(query_act, args.packet_elems)
    total_packets = len(query_packets)
    missing_packets = int(round(total_packets * args.missing_pct / 100.0))
    total_combinations = math.comb(total_packets, missing_packets)
    true_missing = sample_missing_positions(total_packets, missing_packets, args.seed)
    t_prep0 = time.perf_counter()
    observed_packets = drop_packets_preserve_order(query_act, args.packet_elems, true_missing)
    observed_matrix = packets_to_matrix(normalize_packets(observed_packets), args.packet_elems)
    corpus_packet_mats = preprocess_corpus_packets(corpus["raw_activations"], args.packet_elems)
    corpus_signatures = build_corpus_signatures(corpus_packet_mats)
    query_signature = packet_signature(observed_matrix)
    t_prep1 = time.perf_counter()

    t_retr0 = time.perf_counter()
    candidate_indices, signature_sims = select_candidate_indices(
        query_signature,
        corpus_signatures,
        args.top_k_candidates,
    )
    t_retr1 = time.perf_counter()

    results = []
    full_sims = torch.matmul(corpus["activations"].float(), F.normalize(query_act.reshape(-1).float(), dim=0))
    dotprod_sec = 0.0
    dp_sec = 0.0
    for corpus_idx in candidate_indices:
        t_dot0 = time.perf_counter()
        sim = packet_similarity_matrix(observed_matrix, corpus_packet_mats[corpus_idx])
        t_dot1 = time.perf_counter()
        score, aligned_positions, recovered_missing = best_monotonic_alignment(sim)
        t_dp1 = time.perf_counter()
        dotprod_sec += t_dot1 - t_dot0
        dp_sec += t_dp1 - t_dot1
        overlap = len(set(recovered_missing) & set(true_missing))
        exact_match = recovered_missing == true_missing
        results.append(
            {
                "corpus_idx": corpus_idx,
                "score": score,
                "signature_cosine": float(signature_sims[corpus_idx].item()),
                "full_activation_cosine": float(full_sims[corpus_idx].item()),
                "recovered_missing": recovered_missing,
                "aligned_positions": aligned_positions,
                "overlap": overlap,
                "exact_match": exact_match,
                "corpus_path": corpus["paths"][corpus_idx],
                "corpus_label_imagenet": int(corpus["labels_imagenet"][corpus_idx].item()),
            }
        )

    results.sort(key=lambda row: row["score"], reverse=True)
    best = results[0]
    total_search_sec = dotprod_sec + dp_sec
    t_total = t_retr1 - t_edge0 + dotprod_sec + dp_sec + (t_prep1 - t_prep0)

    report_lines = [
        f"query_path={query_path}",
        f"query_label_imagenet={y_imagenet}",
        f"total_packets={total_packets}",
        f"missing_pct={args.missing_pct}",
        f"missing_packets={missing_packets}",
        f"total_combinations={total_combinations}",
        f"candidate_count={len(candidate_indices)}",
        f"true_missing={true_missing}",
        f"best_corpus_idx={best['corpus_idx']}",
        f"best_corpus_path={best['corpus_path']}",
        f"best_corpus_label_imagenet={best['corpus_label_imagenet']}",
        f"best_score={best['score']:.6f}",
        f"best_signature_cosine={best['signature_cosine']:.6f}",
        f"best_full_activation_cosine={best['full_activation_cosine']:.6f}",
        f"recovered_missing={best['recovered_missing']}",
        f"exact_match={best['exact_match']}",
        f"overlap={best['overlap']}/{missing_packets}",
        f"timing_edge_forward_sec={t_edge1 - t_edge0:.6f}",
        f"timing_observed_prep_sec={t_prep1 - t_prep0:.6f}",
        f"timing_candidate_retrieval_sec={t_retr1 - t_retr0:.6f}",
        f"timing_alignment_dotproducts_sec={dotprod_sec:.6f}",
        f"timing_alignment_dp_sec={dp_sec:.6f}",
        f"timing_alignment_total_sec={total_search_sec:.6f}",
        f"timing_script_measured_sec={t_total:.6f}",
    ]

    out_txt = os.path.join(args.out_dir, "single_query_missing_position_search.txt")
    with open(out_txt, "w", encoding="ascii") as f:
        for line in report_lines:
            print(line)
            f.write(line + "\n")
        f.write("\nTop candidates:\n")
        print("Top candidates:")
        for row in results[: args.top_k_report]:
            line = (
                f"corpus_idx={row['corpus_idx']} score={row['score']:.6f} "
                f"sig_cos={row['signature_cosine']:.6f} "
                f"full_cos={row['full_activation_cosine']:.6f} "
                f"exact_match={row['exact_match']} overlap={row['overlap']}/{missing_packets} "
                f"label={row['corpus_label_imagenet']} path={row['corpus_path']}"
            )
            print(line)
            f.write(line + "\n")

    print(f"saved_report={out_txt}")


if __name__ == "__main__":
    main()
