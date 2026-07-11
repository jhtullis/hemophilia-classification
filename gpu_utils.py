import torch


def get_gpu_config(device: torch.device, default_batch_size: int) -> dict:
    """Return training hyperparameters tuned for the detected GPU.

    SM version mapping:
        6.x  — Pascal  (P100):        no Tensor Cores; AMP saves memory only
        7.x  — Volta   (V100):        Tensor Cores; fp16 + GradScaler
        8.x  — Ampere  (A100, L40S):  Tensor Cores; bf16 + compile (needs module load cuda/12.8.1)
        9.x  — Hopper  (H100, H200):  bf16, no GradScaler, compile
        10.x — Blackwell (B200+):     bf16, no GradScaler, compile
    """
    if device.type != "cuda":
        return {
            "amp_enabled": False,
            "amp_dtype":   None,
            "use_scaler":  False,
            "batch_size":  default_batch_size,
            "use_compile": False,
        }

    props = torch.cuda.get_device_properties(device)
    sm    = props.major
    print(f"GPU: {props.name}  (SM {props.major}.{props.minor}, "
          f"{props.total_memory / 1e9:.1f} GB)")

    # cuDNN autotuner: free win when input shapes are fixed (200×200 patches).
    torch.backends.cudnn.benchmark = True

    if sm >= 9:        # Hopper / Blackwell (H100, H200, B200+)
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.bfloat16,
            "use_scaler":  False,
            "batch_size":  64,
            "use_compile": True,
        }
    elif sm >= 8:      # Ampere (A100, L40S, A40, RTX 30xx)
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.bfloat16,
            "use_scaler":  False,   # bf16 on Ampere doesn't need GradScaler
            "batch_size":  64,
            "use_compile": True,    # requires `module load cuda/12.8.1` in Slurm script (e-h)
        }
    elif sm >= 7:      # Volta (V100)
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.float16,
            "use_scaler":  True,
            "batch_size":  64,
            "use_compile": False,
        }
    else:              # Pascal (P100, SM 6.x) — no Tensor Cores
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.float16,
            "use_scaler":  True,
            "batch_size":  32,
            "use_compile": False,
        }
