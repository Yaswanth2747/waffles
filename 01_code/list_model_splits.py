import argparse

from models import SUPPORTED_MODELS, get_num_splits, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print valid split indices for a supported model"
    )
    parser.add_argument(
        "--model-name",
        default="vit_b_16",
        choices=SUPPORTED_MODELS,
        help="Model to inspect",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(model_name=args.model_name, device="cpu")
    num_splits = get_num_splits(model)
    print(f"model_name={args.model_name}")
    print(f"num_feature_blocks={num_splits}")
    print(f"valid_split_indices=0..{num_splits}")


if __name__ == "__main__":
    main()
