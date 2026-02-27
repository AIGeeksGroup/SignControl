# WanControl: ControlNet Integration for Wan2.1 Video Generation

## Overview

WanControl is an extension of the [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) video generation model, an open-source project by Alibaba. This project integrates ​**ControlNet** into the training pipeline of Wan2.1, leveraging the codebase from ​[**DiffSynth-Studio**](https://github.com/modelscope/DiffSynth-Studio). Our ​**ControlNet-Transformer** implementation is inspired by ​[**PIXART-δ**](https://arxiv.org/pdf/2401.05252), which introduces advanced techniques for controllable image and video synthesis. The integration enables fine-grained control over video generation using control signals, such as images or videos.



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shalfun/WanControl.git
   cd WanControl
   ```

2. Install the required dependencies:
   ```bash
   pip install -e .
   ```

## Data Preparation

The dataset should be organized as follows:

```
data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
    ├── video_00001_c.mp4
    ├── image_00002.jpg
    └── image_00002_c.jpg
```


The `metadata.csv` file should contain the following columns:

| Column Name   | Description                     |
|---------------|---------------------------------|
| `file_name`   | Name of the video or image file |
| `text`        | Text description of the file    |
| `control_name`| Name of the control file        |

Example `metadata.csv`:

```
file_name,text,control_name
video_00001.mp4,"video description",video_00001_c.mp4
image_00002.jpg,"image description",image_00002_c.jpg
```



## Model Download
Taking Wan2.1-T2V-1.3B as an example:

Download models using modelscope-cli(Recommended):
```
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir your/model/path/Wan2.1-T2V-1.3B
```
Download models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

Ensure the following checkpoints are available in the specified paths:

- **Text Encoder**: `models_t5_umt5-xxl-enc-bf16.pth`
- **VAE**: `Wan2.1_VAE.pth`
- **DiT Model**: `diffusion_pytorch_model.safetensors`


## Preprocessing

Run the preprocessing script to prepare the data for training:

```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path data/example_dataset \
  --output_path ./models \
  --text_encoder_path "your/model/path/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "your/model/path/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832
```

After preprocessing, the dataset will include `.tensors.pth` files for each video and image:

```
data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
    ├── video_00001_c.mp4
    ├── video_00001.mp4.tensors.pth
    ├── image_00002.jpg
    ├── image_00002_c.jpg
    └── image_00002.jpg.tensors.pth
```

---

## Training

To train the model with ControlNet, run the following command:

```bash
python examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture full \
  --dataset_path data/example_dataset \
  --output_path ./ \
  --dit_path "your/model/path/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 1000 \
  --learning_rate 4e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 8 \
  --control_layers 15
```

> **Note:** When `control_layers` is set to 15 (default value), the overall memory usage is approximately 26G due to most parameters being frozen. If your GPU memory is limited, you may consider reducing `control_layers` (the memory usage is approximately 22G and 19G when set to 10 and 5, respectively).







## Acknowledgments

- **Wan2.1**: Original video generation model by Alibaba.  
- **DiffSynth-Studio**: Codebase used for training and preprocessing.  
- **ControlNet**: Implementation of Controllable Image Generation.  
- **PIXART-δ**: An Implementation of ControlNet under the Dit Architecture.  


