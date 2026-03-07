import argparse
import os

import torch

from ann_index import build_ivf_index, build_signature_matrix
from single_query_missing_position_search import preprocess_corpus_packets


def parse_args():
    parser = argparse.ArgumentParser(description="Build an IVF-style ANN index over corpus packet signatures.")
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--packet-elems", type=int, default=256)
    parser.add_argument("--n-clusters", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="ann_index")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    corpus = torch.load(args.corpus_path, map_location="cpu")
    if "raw_activations" not in corpus:
        raise ValueError("Corpus file missing raw_activations.")

    corpus_packet_mats = preprocess_corpus_packets(corpus["raw_activations"], args.packet_elems)
    signature_matrix = build_signature_matrix(corpus_packet_mats)
    index = build_ivf_index(signature_matrix, n_clusters=args.n_clusters, seed=args.seed)

    out_path = os.path.join(args.out_dir, "ivf_signature_index.pt")
    torch.save(
        {
            "corpus_path": args.corpus_path,
            "packet_elems": args.packet_elems,
            "signature_matrix": signature_matrix,
            "centroids": index["centroids"],
            "assignments": index["assignments"],
            "inverted_lists": index["inverted_lists"],
            "n_clusters": index["n_clusters"],
        },
        out_path,
    )
    print(f"saved_index={out_path}")
    print(f"corpus_size={signature_matrix.shape[0]}")
    print(f"signature_dim={signature_matrix.shape[1]}")
    print(f"n_clusters={index['n_clusters']}")


if __name__ == "__main__":
    main()
