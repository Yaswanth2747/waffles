# Iperf Model Splitter

This workspace is now organized around four top-level directories:

- `01_code/`: all runnable Python code
- `02_data/`: datasets and local test images
- `03_outputs/`: benchmark outputs, corpora, plots, and reports
- `04_notebooks/`: exploratory notebooks kept out of the main code path

The current focus is split inference, packet-loss simulation, and corpus-guided packet-position recovery for `vit_b_16` and `mobilenet_v2`.

## Directory Layout

### `01_code/`
Core and experiment scripts.

Important files:
- `models.py`: pretrained model loading and split counts
- `splitter.py`: edge/server split model wrapper
- `edge_client.py`, `server_node.py`: runtime split-inference path
- `network_udp.py`, `serializer.py`: activation transport helpers
- `activation_corruption.py`: packet zeroing and reconstruction helpers
- `build_activation_corpus.py`: build training activation corpus
- `build_ann_index.py`: build IVF-style ANN index over corpus signatures
- `single_query_packet_ranking.py`: single-query packet-ranking sanity check
- `single_query_missing_position_search.py`: single-query hidden-position recovery
- `benchmark_recovered_position_validation.py`: multi-query recovery benchmark
- `bandwidth_split_validation_report.py`: optimal split vs bandwidth
- `fixed_bw_loss_percent_comparison.py`: fixed-bandwidth fair loss-percentage comparison
- `combine_model_reports.py`: joint ViT/MobileNet comparison figures
- `profiler.py`: layer-wise profiling

### `02_data/`
- `datasets/imagenette2-160/`: dataset used for validation and corpus experiments
- `datasets/imagenette2-160.tgz`: downloaded archive
- `images/`: local test images

### `03_outputs/`
Grouped by experiment family.

- `01_profiling/`: layer-wise model profiling
- `02_validation/`: validation corruption benchmarks
- `03_bandwidth/`: bandwidth-dependent split reports
- `04_comparisons/`: cross-model comparison plots and CSVs
- `05_corpus/`: built corpora, ANN index, single-query corpus experiments
- `06_recovery/`: recovered-position validation results
- `99_archive/`: older smoke and packet-position experiments retained for reference

### `04_notebooks/`
- `model.ipynb`
- `model copy.ipynb`

## Recommended Run Commands

All commands below are run from the repository root.

### Runtime split inference

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 01_code/server_node.py --model-name vit_b_16
```

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
SERVER_IP=127.0.0.1 SERVER_PORT=5005 python3 01_code/edge_client.py --model-name vit_b_16 --split-idx 8 --image-path 02_data/images/test_image.jpg
```

### Build a MobileNetV2 activation corpus

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 01_code/build_activation_corpus.py \
  --data-dir 02_data/datasets/imagenette2-160 \
  --split train \
  --dataset-name imagenette \
  --model-name mobilenet_v2 \
  --split-idx 17 \
  --batch-size 32 \
  --max-samples 5000 \
  --device cuda \
  --out-dir 03_outputs/05_corpus/activation_corpus_mobilenet_5000
```

### Build an IVF ANN index over corpus signatures

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 01_code/build_ann_index.py \
  --corpus-path 03_outputs/05_corpus/activation_corpus_mobilenet_5000/mobilenet_v2_split17_train_corpus.pt \
  --packet-elems 256 \
  --n-clusters 128 \
  --out-dir 03_outputs/05_corpus/ann_index_mobilenet_5000
```

### Recovered-position validation benchmark

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

## Main Results

### 1. ViT vs MobileNet: fixed bandwidth, same packet-loss percentage

Experiment output:
- `03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison/`

At `40 MB/s`, with the same packet-loss percentages for both models:

| Model | Loss % | Clean Top-1 / Top-5 | Oracle Top-1 / Top-5 | Random Top-1 / Top-5 |
|---|---:|---|---|---|
| ViT-B/16 | 20 | 98.0 / 100.0 | 96.0 / 98.0 | 43.3 / 56.7 |
| ViT-B/16 | 40 | 98.0 / 100.0 | 88.0 / 95.0 | 17.7 / 28.7 |
| MobileNetV2 | 20 | 96.0 / 99.0 | 91.0 / 100.0 | 7.3 / 17.3 |
| MobileNetV2 | 40 | 96.0 / 99.0 | 80.0 / 92.0 | 0.3 / 1.7 |

Key point:
- ViT is much more robust to packet-loss uncertainty
- MobileNet is much faster in end-to-end latency

Relevant files:
- `03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison/fixed_bw_accuracy_table.csv`
- `03_outputs/04_comparisons/fixed_bw_40mb_losspct_comparison/accuracy_vs_loss_pct.png`

### 2. Bandwidth-dependent optimal split comparison

Experiment outputs:
- `03_outputs/03_bandwidth/bandwidth_split_validation_report_vit_gpu/`
- `03_outputs/03_bandwidth/bandwidth_split_validation_report_mobilenet_gpu/`
- combined comparison: `03_outputs/04_comparisons/combined_model_comparison/`

Findings:
- ViT-B/16 optimal split stayed constant at `7` across tested bandwidths
- MobileNetV2 optimal split stayed constant at `17` across tested bandwidths
- MobileNetV2 total latency stayed much lower than ViT-B/16 across the bandwidth sweep

Key figures:
- `03_outputs/04_comparisons/combined_model_comparison/combined_publication_figure.png`
- `03_outputs/04_comparisons/combined_model_comparison/combined_total_latency.png`

### 3. Corpus-guided hidden missing-position recovery

The central MobileNetV2 result uses:
- model: `mobilenet_v2`
- split: `17`
- corpus: `5000` training activations
- candidate shortlist: `50`
- validation queries: `100` or `500` depending on experiment

#### Exact/pruned recovery, 100 queries

Outputs:
- `03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k20/`
- `03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k40/`

| Loss % | Exact Match | Mean Overlap | Oracle Top-1 / Top-5 | Recovered Top-1 / Top-5 | Random Top-1 / Top-5 |
|---:|---:|---:|---|---|---|
| 20 | 92.0% | 97.33% | 91.0 / 100.0 | 89.0 / 100.0 | 7.33 / 17.33 |
| 40 | 90.0% | 98.50% | 80.0 / 92.0 | 81.0 / 92.0 | 0.33 / 1.67 |

Interpretation:
- recovered inference closely tracks oracle
- random placement remains catastrophic
- content-based packet alignment is viable in this setting

#### Greedy windowed rematching, 500 queries

Outputs:
- `03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k20/`
- `03_outputs/06_recovery/rematched_validation_mobilenet_500_c5000_k40/`

| Loss % | Exact Match | Mean Overlap | Mean Query Time | Oracle Top-1 / Top-5 | Rematched Top-1 / Top-5 |
|---:|---:|---:|---:|---|---|
| 20 | 77.8% | 88.17% | 0.0261 s | 78.2 / 92.4 | 70.0 / 82.4 |
| 40 | 73.0% | 88.90% | 0.0230 s | 64.0 / 82.4 | 56.8 / 71.6 |

Interpretation:
- very fast
- accuracy drop is too large relative to oracle
- pure greedy local alignment is not enough

#### Banded DP rematching, 500 queries

Outputs:
- `03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k20/`
- `03_outputs/06_recovery/banded_dp_validation_mobilenet_500_c5000_k40/`

| Loss % | Exact Match | Mean Overlap | Mean Query Time | Oracle Top-1 / Top-5 | Rematched Top-1 / Top-5 |
|---:|---:|---:|---:|---|---|
| 20 | 87.0% | 96.87% | 0.3419 s | 78.2 / 92.4 | 77.6 / 92.2 |
| 40 | 80.4% | 96.97% | 0.6846 s | 64.0 / 82.4 | 62.8 / 81.2 |

Interpretation:
- banded DP is the current best compromise
- much better than greedy
- much closer to oracle
- still substantially more practical than the older unconstrained search path

### 4. ANN candidate search test

Output:
- `03_outputs/06_recovery/recovered_position_validation_mobilenet_100_c5000_k20_ann/`

Current verdict:
- IVF-style ANN over packet signatures did not yet beat the exact top-K candidate scan at this scale
- average ANN candidate pool was still relatively large
- ANN is not yet the right optimization target for `5000` corpus items

## Where the bottleneck is now

Single-query timing on the current optimized path showed:
- alignment **dot products** are small (milliseconds)
- the dominant cost is the alignment search logic itself, especially DP-style matching

This means:
- the combinatorial search space is already collapsed by geometry and banding
- the real systems problem is implementation efficiency of the alignment kernel

## Recommended Next Steps

1. Optimize the banded DP implementation further
2. Run the banded DP benchmark on even larger validation slices (`1000+` if practical)
3. Do corpus-size ablation: `100`, `500`, `2000`, `5000`
4. Revisit ANN only when corpus size grows substantially beyond `5000`

## Notes

- `.vscode/` is kept local-editor specific
- `03_outputs/99_archive/` contains older smoke and exploratory outputs that are not part of the mainline narrative
- the notebooks in `04_notebooks/` are preserved but not treated as the authoritative implementation path
