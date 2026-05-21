import torch


def get_gpu_config(device: torch.device, default_batch_size: int) -> dict:
    """Return training hyperparameters tuned for the detected GPU.

    SM version mapping:
        6.x  — Pascal (P100): no Tensor Cores; AMP saves memory only
        7.x  — Volta/Ampere (V100/A100): Tensor Cores; fp16 + GradScaler
        9.x  — Hopper (H200): bf16, no GradScaler, torch.compile
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

    if sm >= 9:        # H200 / H100 (Hopper)
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.bfloat16,
            "use_scaler":  False,
            "batch_size":  64,
            "use_compile": True,
        }
    elif sm >= 7:      # V100 / A100 (Volta / Ampere)
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.float16,
            "use_scaler":  True,
            "batch_size":  64,
            "use_compile": False,
        }
    else:              # P100 (Pascal, SM 6.x) — no Tensor Cores
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.float16,
            "use_scaler":  True,
            "batch_size":  32,
            "use_compile": False,
        }
