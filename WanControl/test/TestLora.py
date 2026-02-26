import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    "/fs/scratch/PFIN0007/ICLR_2025/Sign_World/WanControl/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "/fs/scratch/PFIN0007/ICLR_2025/Sign_World/WanControl/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "/fs/scratch/PFIN0007/ICLR_2025/Sign_World/WanControl/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
], control_layers=15)#这里的control_layers=15是自己加的 , control_layers=15
model_manager.load_lora("/fs/scratch/PFIN0007/ICLR_2025/Sign_World/WanControl/LoRA.ckpt", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

video = pipe(
    prompt="WIND SCHWACH DEUTSCH IX WEHEN WEHEN WEHEN HEUTE NACHT ALPEN DREI GRAD REGION IX DREIZEHN GRAD GRAD __OFF__",
    num_inference_steps=50,
    seed=0, tiled=True
)
save_video(video, "video.mp4", fps=30, quality=5)