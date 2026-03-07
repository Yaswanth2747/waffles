# edge/edge_client.py

import argparse
import socket
import time
import os
import struct
import torch
import requests
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

from activation_corruption import get_total_packets, zero_activation_packets
from models import SUPPORTED_MODELS, load_model
from splitter import SplitModel
from serializer import serialize_tensor, deserialize_tensor
from network_udp import send_data, receive_data


SERVER_IP = os.getenv("SERVER_IP", "192.168.0.109")
PORT = int(os.getenv("SERVER_PORT", "5005"))
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
TIMING_HEADER_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(description="Edge client for split ViT inference")
    parser.add_argument(
        "--model-name",
        default="vit_b_16",
        choices=SUPPORTED_MODELS,
        help="Model to run",
    )
    parser.add_argument("--split-idx", type=int, default=5, help="Layer/block index to split at")
    parser.add_argument("--image-url", type=str, default=DEFAULT_IMAGE_URL, help="Input image URL")
    parser.add_argument("--image-path", type=str, default=None, help="Local image path (overrides --image-url)")
    parser.add_argument(
        "--zero-packets",
        type=int,
        default=0,
        help="Number of activation packets to zero out before transmission",
    )
    parser.add_argument(
        "--packet-elems",
        type=int,
        default=256,
        help="Activation packet size in float elements for zeroing experiment",
    )
    parser.add_argument(
        "--zero-seed",
        type=int,
        default=42,
        help="Random seed for selecting packets to zero",
    )
    parser.add_argument(
        "--target-class-idx",
        type=int,
        default=None,
        help="Optional ImageNet class index to report CE loss",
    )
    parser.add_argument(
        "--no-show-image",
        action="store_true",
        help="Disable image display window",
    )
    return parser.parse_args()


def load_input_tensor(args):
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


def load_imagenet_labels():
    try:
        response = requests.get(LABELS_URL, timeout=20)
        response.raise_for_status()
        labels = [line for line in response.text.splitlines() if line]
        if len(labels) >= 1000:
            return labels
    except Exception as exc:
        print(f"Warning: could not fetch ImageNet labels ({exc}). Using class indices.")

    return [f"class_{i}" for i in range(1000)]


def print_top10(logits, labels):
    probs = F.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=10, dim=1)

    print("Top-10 predictions:")
    for rank in range(10):
        cls_idx = int(top_indices[0, rank].item())
        cls_name = labels[cls_idx] if cls_idx < len(labels) else f"class_{cls_idx}"
        prob = float(top_probs[0, rank].item()) * 100.0
        print(f"{rank+1:2d}. {cls_name:<30} {prob:6.2f}%")


def maybe_show_image(image):
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis("off")
        plt.show()
    except Exception as exc:
        print(f"Warning: could not display image ({exc}).")


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    sock.settimeout(30.0)

    model = load_model(model_name=args.model_name)

    split_idx = args.split_idx

    split = SplitModel(model, split_idx)
    print("Device being used on EDGE : ", next(model.parameters()).device)
    print("Model on EDGE             : ", args.model_name)
    print("Using split_idx on EDGE   : ", split_idx)

    x, input_image = load_input_tensor(args)
    labels = load_imagenet_labels()

    t0 = time.perf_counter()

    act = split.edge_forward(x)
    act, zeroed = zero_activation_packets(
        act,
        packet_elems=args.packet_elems,
        zero_packets=args.zero_packets,
        seed=args.zero_seed,
    )
    if zeroed:
        preview = zeroed[:20]
        extra = "" if len(zeroed) <= 20 else " ..."
        print(
            f"Zeroed activation packets: {len(zeroed)} / "
            f"{get_total_packets(act, args.packet_elems)} "
            f"(packet_elems={args.packet_elems})"
        )
        print(f"Zeroed packet indices (first 20): {preview}{extra}")

    t1 = time.perf_counter()

    payload = struct.pack("I", split_idx) + serialize_tensor(act)

    send_data(sock, payload, (SERVER_IP, PORT))

    try:
        response_payload, _, _ = receive_data(sock)
    except (socket.timeout, TimeoutError):
        print("Timed out waiting for server response.")
        return

    if len(response_payload) < TIMING_HEADER_SIZE:
        print("Invalid response payload: missing timing header.")
        return

    T_comm, T_server = struct.unpack("dd", response_payload[:TIMING_HEADER_SIZE])
    logits = deserialize_tensor(response_payload[TIMING_HEADER_SIZE:])

    T_edge = t1 - t0

    T_total = T_edge + T_comm + T_server

    print("Edge   :", T_edge*1000000   , 'us')
    print("Comm   :", T_comm*1000000   , 'us')
    print("Server :", T_server*1000000 , 'us')
    print("Total  :", T_total*1000000  , 'us')
    print_top10(logits, labels)
    if args.target_class_idx is not None:
        target = torch.tensor([args.target_class_idx], dtype=torch.long)
        loss = F.cross_entropy(logits, target)
        print(f"Cross-entropy loss (target={args.target_class_idx}): {loss.item():.6f}")
    if not args.no_show_image:
        maybe_show_image(input_image)

if __name__ == "__main__":
    main()
