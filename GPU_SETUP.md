# GPU Setup Guide

## Current Status
Your PyTorch installation is **CPU-only**. The code will work but training will be slower.

## Enable GPU Acceleration (NVIDIA GPUs)

### Check if you have an NVIDIA GPU
```powershell
nvidia-smi
```

If this command works, you have an NVIDIA GPU and can use CUDA.

### Install CUDA-enabled PyTorch

**For CUDA 12.1 (Most modern GPUs):**
```powershell
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8 (Older GPUs):**
```powershell
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU is Working
```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}')"
```

## Performance Comparison

| Hardware | Training Speed (100 episodes) |
|----------|------------------------------|
| CPU Only | ~5-10 minutes |
| GPU (CUDA) | ~1-2 minutes |

**Speed improvement: 3-5x faster with GPU**

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in CONFIG (256 → 128 or 64)
- Reduce `replay_buffer_size` (100000 → 50000)
- Close other GPU-using applications

### "CUDA driver version is insufficient"
- Update your NVIDIA GPU drivers from nvidia.com

### AMD GPUs
AMD GPUs are not supported by CUDA. Consider using:
- **ROCm** (AMD's CUDA alternative) - complex setup
- **CPU training** - works fine for this project size
