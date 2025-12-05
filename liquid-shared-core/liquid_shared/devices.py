# liquid_shared/devices.py
"""
Device selection logic for LiquidAI models.
Automatically recommends CPU vs GPU and appropriate dtype based on:
- Available hardware (CUDA availability, VRAM)
- Model size (inferred from model name)
"""
import logging
from typing import Literal, NamedTuple

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "cuda"]
DTypeName = Literal["fp32", "fp16", "bf16"]


class DeviceConfig(NamedTuple):
    """Configuration for model device placement."""
    device: DeviceType
    dtype_name: DTypeName
    torch_dtype: torch.dtype
    reason: str


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except Exception as e:
        logger.warning(f"Could not get GPU memory: {e}")
        return 0.0


def infer_model_size(model_name: str) -> str:
    """
    Infer model size category from model name.
    
    Returns: 'small' (350-700M), 'medium' (1-1.5B), 'large' (2B+), or 'unknown'
    """
    lname = model_name.lower()

    # Check for explicit size markers
    if any(x in lname for x in ["2.6b", "2_6b", "2-6b", "2.5b", "3b"]):
        return "large"
    if any(x in lname for x in ["1.2b", "1_2b", "1-2b", "1b", "1.5b"]):
        return "medium"
    if any(x in lname for x in ["700m", "350m", "500m", "750m"]):
        return "small"

    return "unknown"


def dtype_from_name(name: DTypeName) -> torch.dtype:
    """Convert dtype name to torch.dtype."""
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping.get(name, torch.float32)


def recommend_device(model_name: str) -> DeviceConfig:
    """
    Recommend optimal device and dtype for a model.
    
    Decision logic:
    1. If no CUDA available -> CPU with fp32
    2. If CUDA available:
       - Check GPU memory against model requirements
       - Select appropriate dtype for memory efficiency
       
    Args:
        model_name: HuggingFace model ID or local path
        
    Returns:
        DeviceConfig with device, dtype, and reasoning
    """
    has_cuda = torch.cuda.is_available()

    if not has_cuda:
        return DeviceConfig(
            device="cpu",
            dtype_name="fp32",
            torch_dtype=torch.float32,
            reason="CUDA not available, using CPU with fp32 for stability"
        )

    gpu_mem = get_gpu_memory_gb()
    model_size = infer_model_size(model_name)

    logger.info(f"Detected GPU with {gpu_mem:.1f}GB VRAM, model size: {model_size}")

    # Large models (2.6B+)
    if model_size == "large":
        if gpu_mem >= 10:
            return DeviceConfig(
                device="cuda",
                dtype_name="fp16",
                torch_dtype=torch.float16,
                reason=f"Large model on GPU ({gpu_mem:.1f}GB), using fp16 for memory efficiency"
            )
        else:
            return DeviceConfig(
                device="cpu",
                dtype_name="fp32",
                torch_dtype=torch.float32,
                reason=f"Large model but GPU only has {gpu_mem:.1f}GB, falling back to CPU"
            )

    # Medium models (1.2B)
    if model_size == "medium":
        if gpu_mem >= 6:
            # Prefer bf16 if available for better numerical stability
            if torch.cuda.is_bf16_supported():
                return DeviceConfig(
                    device="cuda",
                    dtype_name="bf16",
                    torch_dtype=torch.bfloat16,
                    reason=f"Medium model on GPU ({gpu_mem:.1f}GB), using bf16"
                )
            return DeviceConfig(
                device="cuda",
                dtype_name="fp16",
                torch_dtype=torch.float16,
                reason=f"Medium model on GPU ({gpu_mem:.1f}GB), using fp16"
            )
        else:
            return DeviceConfig(
                device="cpu",
                dtype_name="fp32",
                torch_dtype=torch.float32,
                reason=f"Medium model but GPU only has {gpu_mem:.1f}GB, falling back to CPU"
            )

    # Small models (700M) or unknown
    if gpu_mem >= 4:
        if torch.cuda.is_bf16_supported():
            return DeviceConfig(
                device="cuda",
                dtype_name="bf16",
                torch_dtype=torch.bfloat16,
                reason=f"Small/unknown model on GPU ({gpu_mem:.1f}GB), using bf16"
            )
        return DeviceConfig(
            device="cuda",
            dtype_name="fp16",
            torch_dtype=torch.float16,
            reason=f"Small/unknown model on GPU ({gpu_mem:.1f}GB), using fp16"
        )

    # Default: CPU for very limited GPU memory
    return DeviceConfig(
        device="cpu",
        dtype_name="fp32",
        torch_dtype=torch.float32,
        reason=f"GPU has limited memory ({gpu_mem:.1f}GB), using CPU for safety"
    )


def check_device_compatibility() -> dict:
    """
    Check and report device compatibility status.
    
    Returns:
        Dict with device info, useful for debugging
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }

    if info["cuda_available"] and info["cuda_device_count"] > 0:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = props.total_memory / (1024**3)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"

    return info


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Device compatibility check:")
    for key, value in check_device_compatibility().items():
        print(f"  {key}: {value}")

    print("\nRecommendations for LiquidAI models:")
    for model in ["LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-2.6B"]:
        config = recommend_device(model)
        print(f"  {model}:")
        print(f"    Device: {config.device}, dtype: {config.dtype_name}")
        print(f"    Reason: {config.reason}")
