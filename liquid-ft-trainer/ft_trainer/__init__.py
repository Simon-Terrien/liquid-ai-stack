# ft_trainer/__init__.py
"""
Liquid Fine-Tuning Trainer - Train LiquidAI models on custom datasets.

Supports:
- Full fine-tuning for smaller models (700M)
- LoRA fine-tuning for medium models (1.2B)
- QLoRA (4-bit) for larger models (2.6B)

Usage:
    from ft_trainer import train, FTConfig
    
    config = FTConfig(
        model_name="LiquidAI/LFM2-700M",
        use_lora=True,
        num_epochs=3,
    )
    
    model_path = train(config, data_path=Path("./data/ft"))
"""

from .train import (
    FTConfig,
    InstructionDataset,
    load_model_for_training,
    train,
)

__version__ = "0.1.0"

__all__ = [
    "train",
    "load_model_for_training",
    "FTConfig",
    "InstructionDataset",
]
