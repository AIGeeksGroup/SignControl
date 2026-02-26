# SignControl: Multi-Granular Control for Sign Language Video Generation

This is the official repository for the paper:
> **SignControl: Multi-Granular Control for Sign Language Video Generation**
>
> [Xuehan Hou]()\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, [Ziye Song]()\*, [Huacan Wang](), and [Zheng Zhu](zhengzhu@ieee.org)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> ### [Paper]() 

SignControl extends the Wan 2.1 T2V 1.3B family with a hierarchical ControlNeXt-based conditioning pipeline that is tailored to the fine-grained needs of sign language video generation. Built on the WanControl implementation, this repository reproduces the three-stage training and inference workflow described in the [SignControl paper](SignControl_paper.pdf): LoRA domain calibration, multi-modal ControlNeXt integration, and Control Decay for robustness under incomplete control.

## Repository Layout

- `SignControl_paper.pdf` – Source paper describing the architecture, experiments, and evaluation on Phoenix-2014.v3.
- `environment.yml` – Conda recipe for `signcontrol-env` (Python 3.9, PyTorch 2.5.1 + CUDA 12.1, DiffSynth, Transformers, etc.).
- `WanControl/` – Local copy of the WanControl project; contains the ControlNeXt extensions, training scripts, and sampling pipelines that SignControl builds upon.
  - `examples/wanvideo/train_wan_t2v.py` – Multi-purpose training script used for LoRA alignment and ControlNeXt tuning.
  - `examples/wanvideo/wan_1.3b_text_to_video.py` – Sample inference pipeline.
  - `requirements.txt` / `setup.py` – Additional Python dependencies for WanControl itself.

## Highlights from the Paper

1. **Hierarchical Multi-Modality Control** – pose (DWPose) for joint-level motion, optical flow (OnlyFlow + PGMM) for pixel gradients, and depth (Depth Anything V2 + Depth LoRA) for coarse spatial layout are injected at progressively deeper DiT blocks (pose → block 1, flow → block 6, depth → block 11). Cross normalization aligns each modality before fusion.
2. **Control Decay Strategy** – after LoRA convergence, modality weights are linearly decayed (with a floor of 0.1) so the model learns to rely on semantics and partial control signals, improving inference robustness when controls are missing.
3. **ControlNeXt on DiT** – the architecture adapts ControlNeXt (originally designed for UNet) to the DiT backbone, letting LoRA adapters fine-tune only attention and FFN layers while ControlNeXt modules provide multi-granular conditioning.
4. **Phoenix-2014.v3 Evaluation** – experiments on ~1,000 weather-forecast sentences show a BLEU of 21.8, ROUGE-L 47.3, and state-of-the-art SSIM / FVD scores compared to SignGen, validating the multi-modal control approach.

## Environment Setup

1. **Prerequisites**
   - CUDA 12.x driver + compatible GPU (4× H20 or equivalent recommended for training).
   - Conda / Miniconda installed.
2. **Create the environment**
   ```bash
   conda env create -f environment.yml
   conda activate signcontrol-env
   ```
   The environment pins Python 3.9.23, PyTorch 2.5.1+cu121, transformers 4.56.1, diffsynth 1.1.2, and NVIDIA CUDA libraries (cuBLAS, cuDNN, cuSPARSE, etc.).
3. **Install WanControl dependencies**
   ```bash
   cd WanControl
   pip install -e .
   ```
   This ensures the ControlNet extensions, DiffSynth integrations, and WanVideo training scripts are discoverable.

## Data Preparation

1. **Phoenix-2014.v3 dataset** – use the standard download (weather-forecast sentences + German annotations). Each video is resized to 480×832 and trimmed to 81 frames, then projected into a latent space ([21, 16, 60, 104]) via the Wan2.1 VAE.
2. **Control Modalities**
   - **Pose** via DWPose (body + face + hand keypoints).
   - **Optical Flow** via OnlyFlow with PGMM to capture pixel velocities.
   - **Depth** via Depth Anything V2 plus Depth LoRA for coarse spatial layout.
3. **Directory layout** (mirrors `WanControl/examples/wanvideo/README.md`):
   ```text
   data/phoenix2014/
   ├── metadata.csv  # file_name,text,control_name
   └── train/
       ├── phoenix_00001.mp4
       ├── phoenix_00001_c.mp4  # control bundle (pose/flow/depth)
       └── phoenix_00001.mp4.tensors.pth
   ```
   Each row in `metadata.csv` should specify the text prompt and the matching control filename (the script will look for all modalities under that control prefix).
4. **Preprocessing**
   Run the data process step to convert videos into `.tensors.pth` and align with LoRA:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python WanControl/examples/wanvideo/train_wan_t2v.py \
     --task data_process \
     --dataset_path data/phoenix2014 \
     --output_path ./preprocessed \
     --text_encoder_path path/to/models_t5_umt5-xxl-enc-bf16.pth \
     --vae_path path/to/Wan2.1_VAE.pth \
     --tiled --num_frames 81 --height 480 --width 832
   ```

## Training Pipeline

SignControl follows the three-stage strategy outlined in the paper.

1. **Stage 1 – LoRA Coarse Alignment**
   - Train only with text prompts to adapt LoRA adapters (q/k/v/o + FFN layers) to sign language semantics.
   - Freeze the ControlNeXt modules; only text embeddings guide the DiT latent updates.
   - Continue until validation loss stabilizes (e.g., 10–20 epochs on Phoenix-2014). This stage provides the baseline for downstream conditioned learning.

2. **Stage 2 – Multi-Modal Control Learning**
   - Load the converged LoRA weights, enable ControlNeXt for each modality, and inject pose/flow/depth features into blocks 1/6/11 with learnable scaling factors.
   - Use CrossNorm to align the modality activation statistics with DiT latents before summing into the transformer blocks.
   - Run the standard WanControl training command to jointly optimize LoRA + ControlNeXt:
     ```bash
     python WanControl/examples/wanvideo/train_wan_t2v.py \
       --task train \
       --train_architecture full \
       --dataset_path data/phoenix2014 \
       --output_path ./checkpoints/stage2 \
       --dit_path path/to/diffusion_pytorch_model.safetensors \
       --steps_per_epoch 500 \
       --max_epochs 1000 \
       --learning_rate 4e-5 \
       --accumulate_grad_batches 1 \
       --use_gradient_checkpointing \
       --dataloader_num_workers 8 \
       --control_layers 15
     ```
   - The `--control_layers` flag determines how many transformer blocks receive ControlNeXt features; the default (15) keeps most weights frozen, lowering memory (~26 GB on a single GPU).

3. **Stage 3 – Control Decay (Robustness Fine-tuning)**
   - Freeze ControlNeXt, continue fine-tuning LoRA while randomly dropping modalities according to a linear decay schedule (`p_m(e)=max(0.1, 1.0−α·max(0,e−e_stable))`).
   - This stage teaches the model to generate plausible sign videos when only a subset of controls (or only text) is available at inference time.
   - Implement the decay by masking control inputs inside the training script or by toggling modality flags sampled from the decay schedule.

## Inference

After training, use the WanControl sampling scripts to generate videos with hierarchical control:

- **Text-only sampling (LoRA-guided)**
  ```bash
  python WanControl/examples/wanvideo/wan_1.3b_text_to_video.py
  ```
  Customize the prompt, negative prompt, and sampling parameters inside the script (or parameterize with CLI flags) to produce single-sentence weather forecasts, news reports, or signer variations.

- **Controlled sampling**
  1. Prepare control artifacts (pose + flow + depth) for a reference video.
  2. Provide the control tensors/frames to the pipeline (the ControlNeXt modules expect them at 81 frames and 3 channels per modality).
  3. Optionally simulate inference uncertainty by masking one or more modalities (e.g., drop depth) to trigger the Control Decay behavior learned during training.

The inference pipeline uses WanControl’s `WanVideoPipeline` (DiffSynth) and can be extended to evaluate metrics such as BLEU, ROUGE-L, SSIM, LPIPS, and FVD against Phoenix-2014 ground truth, matching the paper’s evaluation table.

## Notes

- Keep the `signcontrol-env` environment available when running scripts; the training commands expect PyTorch + GPU-enabled CUDA libraries.
- Hidden or excluded directories such as `signcontrol-env/` are omitted from git to keep the repo lightweight; recreate them via `environment.yml`.
- The full SignControl paper lives in `SignControl_paper.pdf`. Consult section III and the appendix for dataset details, evaluation metrics, and ablation studies on multi-granular control and decay schedules.

