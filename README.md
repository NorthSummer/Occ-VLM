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
<a target="_blank" href="https://github.com/LaVi-Lab">LaVi Lab</a>
<br>
<strong>The Chinese University of Hong Kong</strong>
</div>

---

## Overview

**Occ-VLM** extends Video-3D LLM with explicit 3D geometry priors. By integrating a SigLip-based occupancy prediction network as the vision tower, the model lifts 2D multi-view observations into dense 3D voxel features. World position embeddings inject spatial context, while a grounding head enables fine-grained 3D object localization — all unified within a single VLM.

<p align="center">
    <img src="assets/occvlm.png" width="95%"><br>
</p>

---

## News
- [2025-6] Code & model released.
- [2025-5] Related work **VG-LLM** released — [Paper](https://arxiv.org/abs/2505.24625) | [Code](https://github.com/LaVi-Lab/VG-LLM).

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

## Training

Single-stage full fine-tuning with combined multi-task dataset:

```bash
sh scripts/3d/train/train_multi.sh
```

Key configurable parameters:
- `frame_sampling_strategy`: `uniform`, `mc-ratio90`, `mc-ratio95`
- `frames_upbound`: max frames per video
- `world_position_embedding_type`: `avg-discrete-sin3d`, `avg-discrete-mlp`, etc.
- `ground_head_type`: `infonce`, `mlp`, `score`

## Evaluation

```bash
sh scripts/3d/eval/eval_scanrefer.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_multi3drefer.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_sqa3d.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_scanqa.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
sh scripts/3d/eval/eval_scan2cap.sh <CKPT_NAME> <SAMPLING> <MAX_FRAMES>
```

Add `_lora` suffix for LoRA checkpoints (e.g., `eval_scanrefer_lora.sh`).

## Architecture

| Component | Description |
|---|---|
| Occ Tower | SigLip-based dense 3D occupancy predictor (EmbodiedScan) |
| Vision Resampler | Compresses multi-view features into fixed tokens |
| World Position Embedding | 3D sine or MLP encoding of spatial coordinates |
| LLM | Qwen2-7B-Instruct |
| Ground Head | Contrastive (InfoNCE) head for 3D visual grounding |

## Acknowledgements

- [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT) — base codebase
- [Video-3D LLM](https://github.com/LaVi-Lab/Video-3D-LLM) — multi-view video representation
- [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) — occupancy network & preprocessing
- [ScanNet](http://www.scan-net.org/), [ScanRefer](https://github.com/daveredrum/ScanRefer), [Multi3DRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP), [SQA3D](https://github.com/SilongYong/SQA3D), [ScanQA](https://github.com/ATR-DBI/ScanQA) — datasets

## Citation

If you find this work useful, please consider citing:

```
@misc{zheng2024video3dllmlearningpositionaware,
      title={Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding},
      author={Duo Zheng and Shijia Huang and Liwei Wang},
      year={2024},
      eprint={2412.00493},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00493},
}
@misc{occvlm2025,
      title={Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors},
      author={LaVi Lab},
      year={2025},
      eprint={2505.24625},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.24625},
}
```
