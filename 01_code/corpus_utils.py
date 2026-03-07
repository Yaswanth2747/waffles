import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from bandwidth_split_validation_report import IMAGENETTE_WNID_TO_IDX
from activation_corruption import split_tensor_packets


def infer_dataset_name(data_dir: str, dataset_name: str) -> str:
    if dataset_name != "auto":
        return dataset_name
    return "imagenette" if "imagenette" in data_dir.lower() else "imagenet"


def resolve_split_root(data_dir: str, split: str) -> str:
    candidate = os.path.join(data_dir, split)
    return candidate if os.path.isdir(candidate) else data_dir


def build_target_remap(classes: List[str], dataset_name: str) -> torch.Tensor:
    if dataset_name == "imagenet":
        return torch.tensor(list(range(len(classes))), dtype=torch.long)
    if dataset_name == "imagenette":
        missing = [c for c in classes if c not in IMAGENETTE_WNID_TO_IDX]
        if missing:
            raise ValueError(f"Unsupported Imagenette classes: {missing}")
        return torch.tensor([IMAGENETTE_WNID_TO_IDX[c] for c in classes], dtype=torch.long)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_loader(data_dir: str, split: str, batch_size: int, max_samples: int, dataset_name: str):
    root = resolve_split_root(data_dir, split)
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
    dataset = datasets.ImageFolder(root, transform=transform)
    target_remap = build_target_remap(dataset.classes, dataset_name)
    if max_samples > 0 and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, loader, target_remap


def extract_normalized_activations(split_model, loader, device: str, return_raw: bool = False):
    activations = []
    raw_activations = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            act = split_model.edge_forward(x)
            flat = act.reshape(x.shape[0], -1)
            flat = F.normalize(flat, p=2, dim=1)
            activations.append(flat.cpu())
            if return_raw:
                raw_activations.append(act.cpu())
            labels.append(y.cpu())
    outputs = [torch.cat(activations, dim=0), torch.cat(labels, dim=0)]
    if return_raw:
        outputs.append(torch.cat(raw_activations, dim=0))
    return tuple(outputs)


def get_dataset_paths(dataset) -> List[str]:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base.samples))
    return [base.samples[i][0] for i in indices]


def packet_similarity_matrix(query_act: torch.Tensor, corpus_act: torch.Tensor, packet_elems: int) -> torch.Tensor:
    q_packets = [F.normalize(pkt.float(), dim=0) for pkt in split_tensor_packets(query_act.cpu(), packet_elems)]
    c_packets = [F.normalize(pkt.float(), dim=0) for pkt in split_tensor_packets(corpus_act.cpu(), packet_elems)]

    sim = torch.empty(len(q_packets), len(c_packets))
    for i, q_pkt in enumerate(q_packets):
        for j, c_pkt in enumerate(c_packets):
            common = min(q_pkt.numel(), c_pkt.numel())
            sim[i, j] = torch.dot(q_pkt[:common], c_pkt[:common])
    return sim


def packet_rank_summary(sim: torch.Tensor, valid_positions: List[int]) -> Tuple[float, float, float, float]:
    ranks = []
    for row_idx, true_pos in enumerate(valid_positions):
        row = sim[row_idx]
        order = torch.argsort(row, descending=True)
        rank = int((order == true_pos).nonzero(as_tuple=True)[0].item()) + 1
        ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    top1 = float((ranks_t <= 1).float().mean().item() * 100.0)
    top3 = float((ranks_t <= 3).float().mean().item() * 100.0)
    top5 = float((ranks_t <= 5).float().mean().item() * 100.0)
    mean_rank = float(ranks_t.mean().item())
    return top1, top3, top5, mean_rank
