# Content-Based Packet Alignment for Split Inference

This repository studies split neural inference under packet loss, with a focus on whether missing packet positions can be recovered from the activation content itself.

The core result is simple:

- `random` placement is catastrophic
- `oracle` placement shows the true damage from missing information alone
- `rematched/recovered` placement shows how much of that damage can be removed by alignment

The strongest current result is on `mobilenet_v2` with a training activation corpus of `5000` samples:

- at `20%` packet loss, recovered/rematched inference tracks oracle closely
- at `40%` packet loss, banded DP remains close to oracle while greedy degrades more
- for `vit_b_16`, the model is much more robust to packet-loss uncertainty than MobileNetV2 at the same packet-loss percentage

## Repository Layout

- [01_code](01_code): all runnable Python scripts
- [02_data](02_data): local test images and dataset placeholders
- [03_outputs](03_outputs): plots, CSVs, benchmark outputs
- [04_notebooks](04_notebooks): exploratory notebooks

## Main Figures

### 1. ViT vs MobileNet at fixed bandwidth, same packet-loss percentage

- Figure: [accuracy_vs_loss_pct.png](03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison/accuracy_vs_loss_pct.png)
- Table: [fixed_bw_accuracy_table.csv](03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison/fixed_bw_accuracy_table.csv)

### 2. Bandwidth sweep comparison

- Main multi-panel figure: [combined_publication_figure.png](03_outputs/04_comparisons/combined_model_comparison/combined_publication_figure.png)
- Summary CSV: [combined_summary.csv](03_outputs/04_comparisons/combined_model_comparison/combined_summary.csv)

### 3. Packet-position recovery and rematching on MobileNetV2

- Exact/pruned recovery, 100 queries, 20% loss:
  - [metrics](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k20/recovered_position_inference_metrics.csv)
  - [recovery summary](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k20/recovered_position_recovery_summary.csv)
- Exact/pruned recovery, 100 queries, 40% loss:
  - [metrics](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k40/recovered_position_inference_metrics.csv)
  - [recovery summary](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k40/recovered_position_recovery_summary.csv)
- Greedy rematching, 500 queries:
  - [20% metrics](03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k20/recovered_position_inference_metrics.csv)
  - [40% metrics](03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k40/recovered_position_inference_metrics.csv)
- Banded-DP rematching, 500 queries:
  - [20% metrics](03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k20/recovered_position_inference_metrics.csv)
  - [40% metrics](03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k40/recovered_position_inference_metrics.csv)

## Algorithms

### Split inference and transport

- Model loading: [01_code/models.py](01_code/models.py)
- Split construction: [01_code/splitter.py](01_code/splitter.py)
- Edge runtime: [01_code/edge_client.py](01_code/edge_client.py)
- Server runtime: [01_code/server_node.py](01_code/server_node.py)
- UDP transport: [01_code/network_udp.py](01_code/network_udp.py)
- Tensor serialization: [01_code/serializer.py](01_code/serializer.py)

### Packet corruption and reconstruction helpers

- Packet zeroing and reconstruction: [01_code/activation_corruption.py](01_code/activation_corruption.py)

### Corpus-based recovery pipeline

- Corpus utilities: [01_code/corpus_utils.py](01_code/corpus_utils.py)
- Build training activation corpus: [01_code/build_activation_corpus.py](01_code/build_activation_corpus.py)
- Single-query packet ranking sanity check: [01_code/single_query_packet_ranking.py](01_code/single_query_packet_ranking.py)
- Single-query hidden-position recovery search: [01_code/single_query_missing_position_search.py](01_code/single_query_missing_position_search.py)
- Multi-query recovery benchmark: [01_code/benchmark_recovered_position_validation.py](01_code/benchmark_recovered_position_validation.py)

### Comparison and report generation

- Validation corruption benchmark: [01_code/benchmark_validation.py](01_code/benchmark_validation.py)
- Packet-position validation benchmark: [01_code/benchmark_packet_position_validation.py](01_code/benchmark_packet_position_validation.py)
- Bandwidth/split report: [01_code/bandwidth_split_validation_report.py](01_code/bandwidth_split_validation_report.py)
- Fixed-bandwidth loss-percent comparison: [01_code/fixed_bw_loss_percent_comparison.py](01_code/fixed_bw_loss_percent_comparison.py)
- Combined model comparison plots: [01_code/combine_model_reports.py](01_code/combine_model_reports.py)
- Model split listing: [01_code/list_model_splits.py](01_code/list_model_splits.py)
- Layer profiling: [01_code/profiler.py](01_code/profiler.py)

## Clear Distinction Between Matching Algorithms

All comparisons below are for `mobilenet_v2`, split `17`, corpus size `5000`, top-k candidates `50`.

### A. Oracle

Definition:
- the true missing positions are known
- packets are placed back into the correct slots

Meaning:
- this is the upper bound after packet loss
- it isolates information loss from alignment error

### B. Random

Definition:
- missing positions are unknown
- candidate placements are guessed randomly while preserving order

Meaning:
- this is the failure baseline
- it shows how bad the problem is without alignment

### C. Recovered

Definition:
- positions are hidden
- best corpus candidate and best alignment are searched exactly within the current pruned pipeline

Code:
- [01_code/single_query_missing_position_search.py](01_code/single_query_missing_position_search.py)
- [01_code/benchmark_recovered_position_validation.py](01_code/benchmark_recovered_position_validation.py)

Meaning:
- this is the strongest current recovery result
- it is accurate, but slower than the lighter rematching variants

### D. Greedy Window Rematching

Definition:
- each observed packet only searches a local band of valid positions
- packets are assigned greedily from left to right with monotonicity enforced

Code:
- greedy aligner in [01_code/single_query_missing_position_search.py](01_code/single_query_missing_position_search.py)

Meaning:
- very fast
- loses too much accuracy relative to oracle

### E. Banded DP Rematching

Definition:
- same local band restriction as greedy
- but alignment is solved with dynamic programming over shift states

Code:
- banded-DP aligner in [01_code/single_query_missing_position_search.py](01_code/single_query_missing_position_search.py)

Meaning:
- current best practical compromise
- much closer to oracle than greedy

## Matching Algorithm Results

### Exact/pruned recovered-position search, 100 queries

| Loss % | Mode | Top-1 | Top-5 | CE loss |
|---:|---|---:|---:|---:|
| 20 | clean | 96.0 | 99.0 | 0.2078 |
| 20 | oracle | 91.0 | 100.0 | 0.4352 |
| 20 | recovered | 89.0 | 100.0 | 0.4495 |
| 20 | random | 7.33 | 17.33 | 6.6507 |
| 40 | clean | 96.0 | 99.0 | 0.2078 |
| 40 | oracle | 80.0 | 92.0 | 1.6208 |
| 40 | recovered | 81.0 | 92.0 | 1.6498 |
| 40 | random | 0.33 | 1.67 | 7.7626 |

Recovery quality:

| Loss % | Exact match | Mean overlap fraction |
|---:|---:|---:|
| 20 | 92.0% | 97.33% |
| 40 | 90.0% | 98.50% |

Source files:
- [20% metrics](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k20/recovered_position_inference_metrics.csv)
- [20% recovery](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k20/recovered_position_recovery_summary.csv)
- [40% metrics](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k40/recovered_position_inference_metrics.csv)
- [40% recovery](03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k40/recovered_position_recovery_summary.csv)

### Greedy window rematching, 500 queries

| Loss % | Exact match | Mean overlap | Mean query time | Oracle Top-1 / Top-5 | Rematched Top-1 / Top-5 |
|---:|---:|---:|---:|---|---|
| 20 | 77.8% | 88.17% | 0.0261 s | 78.2 / 92.4 | 70.0 / 82.4 |
| 40 | 73.0% | 88.90% | 0.0230 s | 64.0 / 82.4 | 56.8 / 71.6 |

Source files:
- [20% metrics](03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k20/recovered_position_inference_metrics.csv)
- [20% recovery](03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k20/recovered_position_recovery_summary.csv)
- [40% metrics](03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k40/recovered_position_inference_metrics.csv)
- [40% recovery](03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k40/recovered_position_recovery_summary.csv)

### Banded-DP rematching, 500 queries

| Loss % | Exact match | Mean overlap | Mean query time | Oracle Top-1 / Top-5 | Rematched Top-1 / Top-5 |
|---:|---:|---:|---:|---|---|
| 20 | 87.0% | 96.87% | 0.3419 s | 78.2 / 92.4 | 77.6 / 92.2 |
| 40 | 80.4% | 96.97% | 0.6846 s | 64.0 / 82.4 | 62.8 / 81.2 |

Source files:
- [20% metrics](03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k20/recovered_position_inference_metrics.csv)
- [20% recovery](03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k20/recovered_position_recovery_summary.csv)
- [40% metrics](03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k40/recovered_position_inference_metrics.csv)
- [40% recovery](03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k40/recovered_position_recovery_summary.csv)

### Matching Algorithm Takeaway

| Method | Speed | Accuracy vs oracle | Current verdict |
|---|---|---|---|
| Random | trivial | unusable | baseline only |
| Greedy window | fastest | too much loss | not enough |
| Banded DP | moderate | close to oracle | best current practical method |
| Exact/pruned recovered search | slower | strongest | accuracy reference |

## Model Comparison: ViT-B/16 vs MobileNetV2

### Fixed bandwidth: 40 MB/s, same packet-loss percentage

| Model | Loss % | Clean Top-1 / Top-5 | Oracle Top-1 / Top-5 | Random Top-1 / Top-5 | Total latency |
|---|---:|---|---|---|---:|
| ViT-B/16 | 20 | 98.0 / 100.0 | 96.0 / 98.0 | 43.3 / 56.7 | 29.90 ms |
| ViT-B/16 | 40 | 98.0 / 100.0 | 88.0 / 95.0 | 17.7 / 28.7 | 29.90 ms |
| MobileNetV2 | 20 | 96.0 / 99.0 | 91.0 / 100.0 | 7.3 / 17.3 | 7.58 ms |
| MobileNetV2 | 40 | 96.0 / 99.0 | 80.0 / 92.0 | 0.3 / 1.7 | 7.58 ms |

Interpretation:
- MobileNetV2 is much faster
- ViT-B/16 is much more robust to packet-placement uncertainty

Sources:
- [table CSV](03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison/fixed_bw_accuracy_table.csv)
- [plot](03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison/accuracy_vs_loss_pct.png)

### Bandwidth sweep

Main figure:
- [combined_publication_figure.png](03_outputs/04_comparisons/combined_model_comparison/combined_publication_figure.png)

High-level result:
- tested bandwidth changes end-to-end latency strongly
- for the tested ranges, optimal split stayed fixed for both models
- robustness trends were driven by model and packet-loss mode more than bandwidth itself

Sources:
- [combined summary](03_outputs/04_comparisons/combined_model_comparison/combined_summary.csv)
- [ViT report](03_outputs/03_bandwidth/bandwidth_split_validation_report_vit_gpu/validation_report_by_bandwidth.csv)
- [MobileNet report](03_outputs/03_bandwidth/bandwidth_split_validation_report_mobilenet_gpu/validation_report_by_bandwidth.csv)

## How To Reproduce Main Results

### Banded-DP rematching benchmark

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 01_code/benchmark_recovered_position_validation.py \
  --data-dir 02_data/datasets/imagenette2-160 \
  --dataset-name imagenette \
  --corpus-path 03_outputs/05_corpus/activation_corpus_mobilenet_5000/mobilenet_v2_split17_train_corpus.pt \
  --model-name mobilenet_v2 \
  --split-idx 17 \
  --packet-elems 256 \
  --missing-pct 20 \
  --sample-count 3 \
  --max-samples 500 \
  --aligner banded_dp \
  --top-k-candidates 50 \
  --device cuda \
  --out-dir 03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k20
```

### Fixed-bandwidth comparison

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 01_code/fixed_bw_loss_percent_comparison.py \
  --val-dir 02_data/datasets/imagenette2-160 \
  --dataset-name imagenette \
  --bandwidth-mbps 40 \
  --loss-levels 0,10,20,30,40,50,60 \
  --packet-elems 256 \
  --sample-count 3 \
  --max-samples 100 \
  --device cuda \
  --out-dir 03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison
```

### Combined model report figure

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 01_code/combine_model_reports.py \
  --vit-dir 03_outputs/03_bandwidth/bandwidth_split_validation_report_vit_gpu \
  --mobilenet-dir 03_outputs/03_bandwidth/bandwidth_split_validation_report_mobilenet_gpu \
  --out-dir 03_outputs/04_comparisons/combined_model_comparison
```

## Current Conclusion

The repository now supports a clear claim:

- packet content is sufficient to recover missing packet positions in late MobileNetV2 activations
- random placement is the dominant failure mode
- banded-DP rematching is the best current practical algorithm in this codebase
- ViT-B/16 trades latency for much stronger robustness to packet-placement uncertainty
