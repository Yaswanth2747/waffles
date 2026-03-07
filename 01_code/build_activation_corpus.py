import argparse
import os

import torch

from corpus_utils import build_loader, extract_normalized_activations, get_dataset_paths, infer_dataset_name
from models import SUPPORTED_MODELS, load_model
from splitter import SplitModel


def parse_args():
    parser = argparse.ArgumentParser(description="Build an activation corpus from training data.")
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/ and val/")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--dataset-name", default="auto", choices=["auto", "imagenet", "imagenette"])
    parser.add_argument("--model-name", default="mobilenet_v2", choices=SUPPORTED_MODELS)
    parser.add_argument("--split-idx", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", default="activation_corpus")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    dataset_name = infer_dataset_name(args.data_dir, args.dataset_name)
    dataset, loader, target_remap = build_loader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        dataset_name=dataset_name,
    )

    model = load_model(args.model_name).to(device).eval()
    split_model = SplitModel(model, args.split_idx)
    activations, labels_local, raw_activations = extract_normalized_activations(
        split_model,
        loader,
        device=device,
        return_raw=True,
    )
    labels_imagenet = target_remap[labels_local]
    paths = get_dataset_paths(dataset)

    out_path = os.path.join(
        args.out_dir,
        f"{args.model_name}_split{args.split_idx}_{args.split}_corpus.pt",
    )
    torch.save(
        {
            "model_name": args.model_name,
            "split_idx": args.split_idx,
            "dataset_name": dataset_name,
            "split": args.split,
            "activations": activations,
            "raw_activations": raw_activations,
            "labels_local": labels_local,
            "labels_imagenet": labels_imagenet,
            "paths": paths,
        },
        out_path,
    )
    print(f"Saved corpus: {out_path}")
    print(f"samples={activations.shape[0]} dim={activations.shape[1]}")


if __name__ == "__main__":
    main()
