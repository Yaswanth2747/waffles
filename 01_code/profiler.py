import argparse
import os
import time
from typing import Any, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from models import SUPPORTED_MODELS, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile model layers and generate metric plots"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Profiling device",
    )
    parser.add_argument(
        "--model-name",
        default="vit_b_16",
        choices=SUPPORTED_MODELS,
        help="Model to profile",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (square)",
    )
    parser.add_argument(
        "--out-dir",
        default="profile_outputs",
        help="Directory to save csv/plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots in a window",
    )
    parser.add_argument(
        "--compare-cpu-gpu",
        action="store_true",
        help="Also profile CPU and GPU and save comparison plots",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return "cpu"
    return device_arg


def _iter_tensors(x: Any) -> Iterable[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for item in x:
            yield from _iter_tensors(item)
    elif isinstance(x, dict):
        for item in x.values():
            yield from _iter_tensors(item)


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    for t in _iter_tensors(x):
        return t
    return None


def _numel(x: Any) -> int:
    return sum(t.numel() for t in _iter_tensors(x))


def _tensor_shapes(x: Any) -> str:
    shapes = [list(t.shape) for t in _iter_tensors(x)]
    return str(shapes[:3]) + ("..." if len(shapes) > 3 else "")


def compute_flops(layer: nn.Module, inp: Any, out: Any) -> float:
    in_tensor = _first_tensor(inp)
    out_tensor = _first_tensor(out)

    if in_tensor is None:
        return 0.0

    if isinstance(layer, nn.Conv2d):
        if out_tensor is None or out_tensor.dim() < 4:
            return 0.0
        batch = in_tensor.shape[0]
        out_c = out_tensor.shape[1]
        out_h = out_tensor.shape[2]
        out_w = out_tensor.shape[3]
        kernel_h, kernel_w = layer.kernel_size
        kernel_mul = kernel_h * kernel_w * (layer.in_channels / layer.groups)
        return float(batch * out_c * out_h * out_w * kernel_mul)

    if isinstance(layer, nn.Linear):
        if out_tensor is None:
            return 0.0
        batch_mul = max(1, int(out_tensor.numel() // max(1, out_tensor.shape[-1])))
        return float(batch_mul * layer.in_features * layer.out_features)

    if isinstance(layer, nn.MultiheadAttention):
        q = in_tensor
        if q.dim() != 3:
            return 0.0
        seq_len = q.shape[0]
        batch = q.shape[1]
        embed = q.shape[2]
        # Approximate projections + QK^T + attention*V + output projection.
        proj = 4.0 * batch * seq_len * embed * embed
        attn = 2.0 * batch * seq_len * seq_len * embed
        return proj + attn

    if isinstance(
        layer,
        (
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.Dropout,
            nn.Flatten,
            nn.AdaptiveAvgPool2d,
        ),
    ):
        return float(_numel(out) if _numel(out) > 0 else _numel(inp))

    return 0.0


def profile_model(model: nn.Module, x: torch.Tensor, device: str) -> pd.DataFrame:
    model = model.to(device)
    x = x.to(device)
    model.eval()

    records: List[dict] = []
    hooks = []
    start_times = {}
    layer_idx = 0

    def register_hooks(module: nn.Module):
        nonlocal layer_idx

        if len(list(module.children())) > 0:
            return

        idx = layer_idx
        layer_idx += 1

        def pre_hook(m: nn.Module, inp: Any):
            if device == "cuda":
                torch.cuda.synchronize()
            start_times[id(m)] = time.perf_counter()

        def post_hook(m: nn.Module, inp: Any, out: Any):
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start_times[id(m)]) * 1000.0

            params = sum(p.numel() for p in m.parameters(recurse=False))
            activations = _numel(out)
            flops = compute_flops(m, inp, out)

            records.append(
                {
                    "layer": idx,
                    "name": m.__class__.__name__,
                    "params": int(params),
                    "param_MB": params * 4.0 / (1024.0**2),
                    "activations": int(activations),
                    "activation_MB": activations * 4.0 / (1024.0**2),
                    "flops": float(flops),
                    "time_ms": float(elapsed_ms),
                    "in_shape": _tensor_shapes(inp),
                    "out_shape": _tensor_shapes(out),
                }
            )

        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_forward_hook(post_hook))

    model.apply(register_hooks)

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    df = pd.DataFrame(records).sort_values("layer").reset_index(drop=True)
    df["cum_params"] = df["params"].cumsum()
    df["cum_param_MB"] = df["param_MB"].cumsum()
    df["cum_activation_MB"] = df["activation_MB"].cumsum()
    df["cum_flops"] = df["flops"].cumsum()
    df["cum_time"] = df["time_ms"].cumsum()
    return df


def plot_layer_metrics(df: pd.DataFrame, out_dir: str, suffix: str, show: bool):
    layers = df["layer"]

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    ax[0, 0].bar(layers, df["params"])
    ax[0, 0].set_title("Parameters Per Layer")
    ax[0, 0].set_ylabel("Count")

    ax[0, 1].bar(layers, df["activation_MB"])
    ax[0, 1].set_title("Activation Size Per Layer")
    ax[0, 1].set_ylabel("MB")

    ax[1, 0].bar(layers, df["flops"])
    ax[1, 0].set_title("FLOPs Per Layer")
    ax[1, 0].set_ylabel("FLOPs")

    ax[1, 1].bar(layers, df["time_ms"])
    ax[1, 1].set_title("Time Per Layer")
    ax[1, 1].set_ylabel("ms")

    for a in ax.flat:
        a.set_xlabel("Layer Index")
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, f"layer_metrics_{suffix}.png")
    fig.savefig(p, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_cumulative_metrics(df: pd.DataFrame, out_dir: str, suffix: str, show: bool):
    layers = df["layer"]
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    ax[0, 0].plot(layers, df["cum_params"], linewidth=2)
    ax[0, 0].set_title("Cumulative Parameters")
    ax[0, 0].set_ylabel("Count")

    ax[0, 1].plot(layers, df["cum_activation_MB"], linewidth=2)
    ax[0, 1].set_title("Cumulative Activation Size")
    ax[0, 1].set_ylabel("MB")

    ax[1, 0].plot(layers, df["cum_flops"], linewidth=2)
    ax[1, 0].set_title("Cumulative FLOPs")
    ax[1, 0].set_ylabel("FLOPs")

    ax[1, 1].plot(layers, df["cum_time"], linewidth=2)
    ax[1, 1].set_title("Cumulative Time")
    ax[1, 1].set_ylabel("ms")

    for a in ax.flat:
        a.set_xlabel("Layer Index")
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, f"cumulative_metrics_{suffix}.png")
    fig.savefig(p, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_cpu_gpu_comparison(df_cpu: pd.DataFrame, df_gpu: pd.DataFrame, out_dir: str, show: bool):
    n = min(len(df_cpu), len(df_gpu))
    if n == 0:
        return

    df_cpu = df_cpu.iloc[:n]
    df_gpu = df_gpu.iloc[:n]
    layers = df_cpu["layer"]

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ax[0, 0].bar(layers, df_cpu["params"])
    ax[0, 0].set_title("Params Per Layer")

    ax[0, 1].bar(layers, df_cpu["activation_MB"])
    ax[0, 1].set_title("Activation MB Per Layer")

    ax[0, 2].bar(layers, df_cpu["flops"])
    ax[0, 2].set_title("FLOPs Per Layer")

    ax[1, 0].plot(layers, df_cpu["cum_activation_MB"], label="CPU")
    ax[1, 0].plot(layers, df_gpu["cum_activation_MB"], label="GPU")
    ax[1, 0].set_title("Cumulative Activation MB")
    ax[1, 0].legend()

    ax[1, 1].plot(layers, df_cpu["cum_flops"], label="CPU")
    ax[1, 1].plot(layers, df_gpu["cum_flops"], label="GPU")
    ax[1, 1].set_title("Cumulative FLOPs")
    ax[1, 1].legend()

    ax[1, 2].plot(layers, df_cpu["cum_time"], label="CPU")
    ax[1, 2].plot(layers, df_gpu["cum_time"], label="GPU")
    ax[1, 2].set_title("Cumulative Time")
    ax[1, 2].legend()

    for a in ax.flat:
        a.set_xlabel("Layer Index")
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "cpu_gpu_comparison.png")
    fig.savefig(p, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def summarize(df: pd.DataFrame, device: str):
    total_params = int(df["params"].sum())
    total_flops = float(df["flops"].sum())
    total_act_mb = float(df["activation_MB"].sum())
    total_time_ms = float(df["time_ms"].sum())
    print(f"[{device}] layers: {len(df)}")
    print(f"[{device}] total params: {total_params:,}")
    print(f"[{device}] total flops (approx): {total_flops:,.0f}")
    print(f"[{device}] total activation MB (summed): {total_act_mb:,.2f}")
    print(f"[{device}] forward time ms (summed hooks): {total_time_ms:,.2f}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(model_name=args.model_name, device="cpu")
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size)

    main_device = resolve_device(args.device)
    print(f"Model: {args.model_name}")
    print(f"Profiling device: {main_device}")
    df_main = profile_model(model, x, main_device)
    summarize(df_main, main_device)

    csv_main = os.path.join(args.out_dir, f"profile_{main_device}.csv")
    df_main.to_csv(csv_main, index=False)
    plot_layer_metrics(df_main, args.out_dir, main_device, args.show)
    plot_cumulative_metrics(df_main, args.out_dir, main_device, args.show)

    if args.compare_cpu_gpu:
        df_cpu = profile_model(model, x, "cpu")
        df_cpu.to_csv(os.path.join(args.out_dir, "profile_cpu.csv"), index=False)

        if torch.cuda.is_available():
            df_gpu = profile_model(model, x, "cuda")
            df_gpu.to_csv(os.path.join(args.out_dir, "profile_gpu.csv"), index=False)
            plot_cpu_gpu_comparison(df_cpu, df_gpu, args.out_dir, args.show)
            summarize(df_cpu, "cpu")
            summarize(df_gpu, "cuda")
        else:
            print("Skipping GPU comparison: CUDA unavailable.")

    print(f"Saved profiler outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
