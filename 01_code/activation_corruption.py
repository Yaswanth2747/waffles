import math

import torch


def get_total_packets(tensor: torch.Tensor, packet_elems: int) -> int:
    if packet_elems <= 0:
        raise ValueError("packet_elems must be > 0")
    return math.ceil(tensor.numel() / packet_elems)


def zero_activation_packets(
    act: torch.Tensor,
    packet_elems: int,
    zero_packets: int,
    seed: int,
):
    if zero_packets <= 0:
        return act, []
    if packet_elems <= 0:
        raise ValueError("packet_elems must be > 0")

    act_corrupt = act.clone()
    flat = act_corrupt.view(-1)
    total_packets = get_total_packets(act_corrupt, packet_elems)
    if total_packets == 0:
        return act_corrupt, []

    zero_packets = min(zero_packets, total_packets)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    selected = torch.randperm(total_packets, generator=gen)[:zero_packets].tolist()
    selected.sort()

    for pkt_idx in selected:
        start = pkt_idx * packet_elems
        end = min(start + packet_elems, flat.numel())
        flat[start:end] = 0.0

    return act_corrupt, selected


def get_packet_lengths(numel: int, packet_elems: int):
    if packet_elems <= 0:
        raise ValueError("packet_elems must be > 0")
    full_packets = numel // packet_elems
    remainder = numel % packet_elems
    lengths = [packet_elems] * full_packets
    if remainder:
        lengths.append(remainder)
    return lengths


def split_tensor_packets(tensor: torch.Tensor, packet_elems: int):
    flat = tensor.reshape(-1)
    lengths = get_packet_lengths(flat.numel(), packet_elems)
    packets = []
    start = 0
    for length in lengths:
        end = start + length
        packets.append(flat[start:end].clone())
        start = end
    return packets


def sample_missing_positions(total_packets: int, missing_packets: int, seed: int):
    if missing_packets < 0:
        raise ValueError("missing_packets must be >= 0")
    missing_packets = min(missing_packets, total_packets)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    selected = torch.randperm(total_packets, generator=gen)[:missing_packets].tolist()
    selected.sort()
    return selected


def drop_packets_preserve_order(tensor: torch.Tensor, packet_elems: int, missing_positions):
    packets = split_tensor_packets(tensor, packet_elems)
    missing_set = set(missing_positions)
    observed_packets = [
        packet.clone() for idx, packet in enumerate(packets) if idx not in missing_set
    ]
    return observed_packets


def reconstruct_from_observed_packets(
    observed_packets,
    original_shape,
    packet_elems: int,
    missing_positions,
    dtype: torch.dtype,
    device,
):
    numel = math.prod(original_shape)
    lengths = get_packet_lengths(numel, packet_elems)
    total_packets = len(lengths)
    missing_set = set(missing_positions)

    if len(observed_packets) != total_packets - len(missing_set):
        raise ValueError("Observed packet count does not match missing positions")

    flat = torch.zeros(numel, dtype=dtype, device=device)
    obs_idx = 0
    cursor = 0

    for pkt_idx, length in enumerate(lengths):
        end = cursor + length
        if pkt_idx not in missing_set:
            packet = observed_packets[obs_idx].to(device=device, dtype=dtype)
            copy_len = min(length, packet.numel())
            flat[cursor:cursor + copy_len] = packet[:copy_len]
            obs_idx += 1
        cursor = end

    return flat.reshape(original_shape)
