# server/server_node.py

import socket
import time
import os
import struct
import argparse

from serializer import deserialize_tensor, serialize_tensor
from network_udp import receive_data, send_data
from models import SUPPORTED_MODELS, load_model
from splitter import SplitModel
import torch



PORT = int(os.getenv("SERVER_PORT", "5005"))
SPLIT_HEADER_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Server node for split inference")
    parser.add_argument(
        "--model-name",
        default="vit_b_16",
        choices=SUPPORTED_MODELS,
        help="Model to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.settimeout(30.0)

    sock.bind(("", PORT))

    model = load_model(model_name=args.model_name)

    device = 'cpu'
    if torch.cuda.is_available() : 
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        model = model.to('cuda')    
    split_cache = {}


    print(f'Using device : {device}')
    print(f'Model on SERVER : {args.model_name}')
    print("Server ready")

    while True:

        print("Waiting for activation...")

        try:
            data, addr, T_comm = receive_data(sock)
        except TimeoutError as exc:
            print(f"Receive timeout: {exc}")
            continue

        print("Activation received")

        if len(data) < SPLIT_HEADER_SIZE:
            print("Invalid payload: missing split header")
            continue

        split_idx = struct.unpack("I", data[:SPLIT_HEADER_SIZE])[0]
        activation = deserialize_tensor(data[SPLIT_HEADER_SIZE:])
        print(f"Using split_idx on SERVER : {split_idx}")
        zero_ratio = (activation == 0).float().mean().item() * 100.0
        print(f"Activation exact-zero ratio : {zero_ratio:.4f}%")

        if split_idx not in split_cache:
            split_cache[split_idx] = SplitModel(model, split_idx)
        split = split_cache[split_idx]

        if torch.cuda.is_available() : activation = activation.to('cuda')

        t0 = time.perf_counter()
        out = split.server_forward(activation)
        t1 = time.perf_counter()

        T_server = t1 - t0

        response = struct.pack("dd", T_comm, T_server) + serialize_tensor(out)

        print("Sending results back")

        send_data(sock, response, addr)

        print("Done\n")


if __name__ == "__main__":
    main()
