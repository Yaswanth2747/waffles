from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from single_query_missing_position_search import build_corpus_signatures


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=1)


def build_signature_matrix(corpus_packet_mats):
    return normalize_rows(build_corpus_signatures(corpus_packet_mats))


def build_ivf_index(signature_matrix: torch.Tensor, n_clusters: int, seed: int):
    x = signature_matrix.cpu().numpy().astype(np.float32)
    n_clusters = min(n_clusters, x.shape[0])
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(256, x.shape[0])),
        n_init="auto",
    )
    assignments = kmeans.fit_predict(x)
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    centroids = normalize_rows(centroids)

    inverted_lists = defaultdict(list)
    for idx, cid in enumerate(assignments.tolist()):
        inverted_lists[int(cid)].append(idx)

    return {
        "centroids": centroids,
        "assignments": torch.tensor(assignments, dtype=torch.long),
        "inverted_lists": dict(inverted_lists),
        "n_clusters": n_clusters,
    }


def retrieve_ivf_candidates(
    query_signature: torch.Tensor,
    signature_matrix: torch.Tensor,
    index,
    top_k: int,
    nprobe: int,
):
    query_signature = F.normalize(query_signature.float(), p=2, dim=0)
    centroid_scores = torch.mv(index["centroids"], query_signature)
    nprobe = min(nprobe, centroid_scores.numel())
    probe_ids = torch.topk(centroid_scores, k=nprobe).indices.tolist()

    candidate_set = []
    seen = set()
    for cid in probe_ids:
        for idx in index["inverted_lists"].get(int(cid), []):
            if idx not in seen:
                seen.add(idx)
                candidate_set.append(idx)

    if not candidate_set:
        sims = torch.mv(signature_matrix, query_signature)
        if top_k and top_k > 0 and top_k < sims.numel():
            return torch.topk(sims, k=top_k).indices.tolist(), sims, 0
        return list(range(sims.numel())), sims, 0

    candidate_tensor = torch.tensor(candidate_set, dtype=torch.long)
    sims_subset = torch.mv(signature_matrix[candidate_tensor], query_signature)
    if top_k and top_k > 0 and top_k < sims_subset.numel():
        top_local = torch.topk(sims_subset, k=top_k).indices
        chosen = candidate_tensor[top_local]
    else:
        order = torch.argsort(sims_subset, descending=True)
        chosen = candidate_tensor[order]

    full_sims = torch.full((signature_matrix.shape[0],), float("-inf"), dtype=torch.float32)
    full_sims[candidate_tensor] = sims_subset
    return chosen.tolist(), full_sims, len(candidate_set)
