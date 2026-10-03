<div align="center">

<h1>ReLIQS</h1>

<h3>Learning Where to Look and How to Judge</h3>

<p>Resolution-agnostic Image Quality Assessment with Quality-aware Saliency</p>

<p>
  <strong>Hakan Emre Gedik · Shashank Gupta · Alan Bovik</strong><br>
  The University of Texas at Austin · University of Colorado Boulder
</p>

<p><strong>Official implementation · CVPR 2026</strong></p>

<p>
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Gedik_Learning_Where_to_Look_and_How_to_Judge_Resolution-agnostic_Image_CVPR_2026_paper.pdf"><img src="https://img.shields.io/badge/Paper-CVPR_2026-2563EB?style=flat-square" alt="Paper · CVPR 2026"></a>
  <a href="https://openaccess.thecvf.com/content/CVPR2026/html/Gedik_Learning_Where_to_Look_and_How_to_Judge_Resolution-agnostic_Image_CVPR_2026_paper.html"><img src="https://img.shields.io/badge/CVF-Project_Page-475569?style=flat-square" alt="CVF project page"></a>
  <a href="https://huggingface.co/hakanemre/ReLIQS"><img src="https://img.shields.io/badge/Hugging_Face-Model_Weights-FFD21E?style=flat-square&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face model weights"></a>
</p>

<p>
  <a href="#installation">Installation</a> ·
  <a href="#image-quality-prediction">Inference</a> ·
  <a href="#saliency-visualization">Saliency</a> ·
  <a href="#training">Training</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

---

## Overview

**ReLIQS** (**Re**solution-agnostic **L**earning for **I**mage **Q**uality with **S**aliency) predicts image quality without a reference image. It combines multiscale patches, CLIP features, quality-aware saliency, and latent quality axes into a single quality score.

- **Resolution-agnostic:** preserves original-resolution cues through patch-based processing.
- **Quality-aware:** learns spatial importance from image-quality supervision.
- **Joint training:** combines datasets through within-dataset ranking and correlation losses.
- **Ready to use:** image scoring, saliency visualization, and distributed training.

<p align="center">
  <img src="assets/overview.png" alt="ReLIQS pipeline: multiscale patch sampling, quality-aware saliency, and latent quality axes" width="100%">
  <br>
  <em>Figure 1. Overview of ReLIQS.</em>
</p>

## Installation

```bash
git clone https://github.com/hakan-emre-gedik/ReLIQS.git
cd ReLIQS

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install `torch` and `torchvision` using the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your Python version and CUDA driver, then install the remaining dependencies:

```bash
python -m pip install numpy scipy pandas pillow opencv-python \
    matplotlib plotly timm ftfy regex tqdm packaging huggingface_hub
```

**Training:** Linux, NVIDIA GPUs with bfloat16 support, and PyTorch with NCCL, `torch.amp`, and `init_process_group(device_id=...)`. **Inference:** automatically uses CUDA when available, otherwise CPU.

CLIP and TinyCLIP are bundled in `clip/` and `tiny_clip/`. Their backbone weights download on first use, including when loading a ReLIQS checkpoint. For offline use, populate `~/.cache/clip` on each machine beforehand.

> Run all commands from the repository root.

## Pretrained Weights

Download a checkpoint from **[Hugging Face](https://huggingface.co/hakanemre/ReLIQS)**. Save it as `model_weights.pth` or specify its path with `--checkpoint_path`.

Both inference scripts use EMA weights: image scoring requires `model_ema` and `patch_sampler_ema`; saliency visualization requires `patch_sampler_ema`.

## Image Quality Prediction

```bash
python score_image.py \
    --checkpoint_path model_weights.pth \
    --image_path /path/to/image.jpg
```

Prints `Score: ...` in **[0, 1]**, where **higher is better**. Scores are not calibrated to a dataset's original MOS scale.

## Saliency Visualization

```bash
python output_saliency.py \
    --checkpoint_path model_weights.pth \
    --image_path /path/to/image.jpg
```

Saves `saliency.png` at the input image's resolution, overwriting any existing file. The map is normalized to 8-bit grayscale; brighter regions indicate greater learned importance, not local quality or distortion severity.

> **Shared defaults:** both scripts use `model_weights.pth`, `image.jpg`, and `options/main_training/koniq.json`. Change `CONFIG_PATH` in the scripts for different model settings; neither accepts `--config`. Dataset files are not required for inference.

## Dataset Preparation

Split annotations are provided in `vlm_iqa_datasets/`. Download images from the corresponding dataset sources and place each prepared dataset at the loader's expected location:

```text
../vlm_iqa_datasets/<dataset_name>/
```

If keeping datasets inside the repository, update the loader path or create a symlink at `../vlm_iqa_datasets`.

For KonIQ, paths relative to the repository root are:

| Path | Contents |
| --- | --- |
| `../vlm_iqa_datasets/KONIQ/images/` | Images |
| `../vlm_iqa_datasets/KONIQ/splits/1/train.txt` | Training annotations |
| `../vlm_iqa_datasets/KONIQ/splits/1/val.txt` | Validation annotations |
| `../vlm_iqa_datasets/KONIQ/splits/1/test.txt` | Test annotations |

The configuration's `split` selects the directory under `splits/`. Annotation filenames must contain `train`, `val`, or `test`.

**Dataset selection:** keep `data.dataset_names` and `data.samples` aligned. Each positive `samples` entry sets that dataset's per-GPU training batch size. A zero entry disables training on it but retains evaluation. Every listed dataset requires train, validation, and test annotations.

## Training

`train.sh` launches `torchrun` with one process per GPU and supports one or multiple machines.

### One machine

Train on four GPUs by default:

```bash
bash train.sh options/main_training/koniq.json runs/koniq
```

Select two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 \
    bash train.sh options/main_training/koniq.json runs/koniq_2gpu
```

For one GPU, set `GPUS_PER_NODE=1`.

### Multiple machines

Run the following on **every machine**, with a unique `NODE_RANK` from `0` to `NNODES - 1`. This example uses four machines with one GPU each:

```bash
NNODES=4 \
GPUS_PER_NODE=1 \
NODE_RANK=0 \
MASTER_ADDR=10.0.0.1 \
MASTER_PORT=29500 \
bash train.sh options/main_training/koniq.json runs/koniq
```

Use the rank-0 machine's reachable address for `MASTER_ADDR`. All machines must share launch settings except `NODE_RANK`, use matching code and configurations, and have access to the same datasets and splits. Launch manually or through a scheduler/SSH launcher; `torchrun` does not start remote commands.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NNODES` | `1` | Number of machines |
| `GPUS_PER_NODE` | `4` | GPU processes per machine |
| `NODE_RANK` | `0` on one machine | Unique machine index; required for multiple machines |
| `MASTER_ADDR` | `127.0.0.1` on one machine | Rank-0 address; required for multiple machines |
| `MASTER_PORT` | `29500` | Coordination port; use distinct ports for concurrent jobs |

### Batch sizes

```text
global_batch_size = samples[dataset_index] * NNODES * GPUS_PER_NODE
```

KonIQ defaults to **16 images per GPU × 4 GPUs = 64 images**. To preserve this global batch size, set KonIQ's entry in `samples` to 32 on two GPUs or 64 on one, subject to available memory.

Ranking and correlation losses use predictions gathered across GPUs. Changing the global batch size changes these losses; ordinary gradient accumulation is not equivalent.

### Configurations

Configurations are in `options/main_training/`. The table lists training datasets; all entries in `dataset_names` are evaluated.

| Configuration | Training datasets |
| --- | --- |
| `koniq.json` | KonIQ |
| `koniq_spaq_kadid.json` | KonIQ, SPAQ, KADID |
| `koniq_spaq_kadid_live_csiq_bid.json` | KonIQ, SPAQ, KADID, LIVE, CSIQ, BID, CLIVE (`ChallengeDB_release`) |
| `koniq_spaq_uhd.json` | KonIQ, SPAQ, UHD |

<details>
<summary><strong>Configuration reference</strong></summary>

| Setting | Default | Meaning |
| --- | --- | --- |
| `split` | `1` | Dataset split directory |
| `init_epochs` | `16` | Initial epochs with both vision encoders frozen |
| `epochs` | `64` | End-to-end training epochs |
| `lr` / `weight_decay` | `1e-5` / `1e-3` | AdamW settings |
| `scheduler_T_max` | `5` | Cosine scheduler setting |
| `train_num_patches` | `[6, 5, 1]` | Training patches per scale, largest to smallest |
| `patch_size` / `num_scales` | `224` / `3` | Patch dimensions and scale count |
| `min_short_edge` | `224` | Smallest-scale short edge |
| `patch_stride` | `0.75` | Evaluation grid density; smaller means more patches |
| `model_name` / `num_attributes` | `ViT-B/16` / `4` | CLIP backbone and latent quality axes |

</details>

Scale short edges are spaced linearly between the input short edge and `min_short_edge`. Evaluation uses a deterministic patch grid with learned pooling weights. Top-k selection is not enabled; `test_num_patches` and `pool_num_patches` do not control patch counts in this execution path.

## Evaluation and Outputs

`main.py` evaluates primary and EMA models after initialization and each end-to-end epoch, reporting **SRCC** and **PLCC** on the configured test datasets.

| Output under the run directory | Contents |
| --- | --- |
| `checkpoints/checkpoint_-1.pth` | Weights after initialization |
| `checkpoints/checkpoint_<epoch>.pth` | Epoch checkpoints, starting at 0 |
| `results/log.csv` | Learning rates and evaluation metrics |
| `results/Plots.html` | Interactive metric plots |
| `train_node_<rank>.log` | Per-machine console output |

Checkpoints contain `model`, `model_ema`, `patch_sampler`, and `patch_sampler_ema`. They store weights only; full training resumption is not implemented. Global rank 0 writes checkpoints and evaluation results.

Score an image with a trained checkpoint:

```bash
python score_image.py \
    --checkpoint_path runs/koniq/checkpoints/checkpoint_63.pth \
    --image_path /path/to/image.jpg
```

Dataset evaluation runs within training; no separate evaluation CLI is provided. Keep splits and preprocessing consistent across experiments and select checkpoints using validation data.

## Citation

If you use ReLIQS in your research, please cite:

```bibtex
@InProceedings{Gedik_2026_CVPR,
    author    = {Gedik, Hakan Emre and Gupta, Shashank and Bovik, Alan},
    title     = {Learning Where to Look and How to Judge: Resolution-agnostic Image Quality Assessment with Quality-aware Saliency},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {37507-37517}
}
```
