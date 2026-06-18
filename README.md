# Occ-VLM: Occupancy-Enhanced Vision-Language Model for 3D Scene Understanding


<div align="center">
    <a href="https://arxiv.org/abs/2505.24625" target="_blank">
    <img src="https://img.shields.io/badge/VG--LLM-ArXiv-red" alt="Paper arXiv"></a>
    <a href="https://github.com/LaVi-Lab/VG-LLM" target="_blank">
    <img src="https://img.shields.io/badge/VG--LLM-Code-blue" alt="Code"></a>
    <a href="https://huggingface.co/datasets/zd11024/Video-3D-LLM_data" target="_blank">
    <img src="https://img.shields.io/badge/Video--3D_LLM-data-green" alt="Data"></a>
    <a href="https://huggingface.co/zd11024/Video3D-LLM-LLaVA-Qwen-Uniform-32" target="_blank">
    <img src="https://img.shields.io/badge/Video--3D_LLM-model-orange" alt="Model"></a>
</div>

<div align="center">
<a>Jianing Li</a>,
<a>Zhou Fang</a>,
<a>Yijiang Liu</a> and
<a>Li Du</a>

<br>
<strong>School of Electronic Science and Engineering, Nanjing University</strong>
</div>

---

## Overview

**Occ-VLM** extends Video-3D LLM with explicit 3D geometry priors. By integrating a SigLip-based occupancy prediction network as the vision tower, the model lifts 2D multi-view observations into dense 3D voxel features. World position embeddings inject spatial context, while a grounding head enables fine-grained 3D object localization — all unified within a single VLM.

---

## News
- [2025-6] Code & model released. — [Model]([Electronics/occ3dllm-LLaVA-Qwen-video](https://huggingface.co/Electronics/occ3dllm-LLaVA-Qwen-video)).

## Supported Tasks

| Task | Type |
|---|---|
| [ScanRefer](https://github.com/daveredrum/ScanRefer) | 3D Visual Grounding |
| [Multi3DRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) | Multi-Object 3D Grounding |
| [SQA3D](https://github.com/SilongYong/SQA3D) | 3D Situated QA |
| [ScanQA](https://github.com/ATR-DBI/ScanQA) | 3D Scene QA |
| [Scan2Cap](https://github.com/daveredrum/ScanRefer) | 3D Dense Captioning |

## Installation

```bash
git clone https://github.com/LaVi-Lab/Occ-VLM.git
cd Occ-VLM
conda create -n occvlm python=3.10 -y && conda activate occvlm
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation

See [data preprocessing guide](scripts/3d/preprocessing/README.md) for full instructions. Quick steps:

1. Download [ScanNet v2](http://www.scan-net.org/)
2. Download [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan)
3. Extract images & point clouds via provided scripts
4. Process downstream annotations (ScanRefer, SQA3D, ScanQA, etc.)



## Evaluation

```bash
sh scripts/3d/eval/eval_scanrefer.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_multi3drefer.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_sqa3d.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_scanqa.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_scan2cap.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
```

Add `_lora` suffix for LoRA checkpoints (e.g., `eval_scanrefer_lora.sh`).

## Acknowledgements

- [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT) — base codebase
- [Video-3D LLM](https://github.com/LaVi-Lab/Video-3D-LLM) — multi-view video representation
- [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) — occupancy network & preprocessing
- [ScanNet](http://www.scan-net.org/), [ScanRefer](https://github.com/daveredrum/ScanRefer), [Multi3DRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP), [SQA3D](https://github.com/SilongYong/SQA3D), [ScanQA](https://github.com/ATR-DBI/ScanQA) — datasets


