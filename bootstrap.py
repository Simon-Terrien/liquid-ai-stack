#!/usr/bin/env python3
"""
Bootstrap script for LiquidAI Stack

Downloads LiquidAI models from HuggingFace and sets up the environment.

Usage:
    python bootstrap.py                  # Download all models (2.6B, 1.2B, 700M)
    python bootstrap.py --model 700M     # Download only 700M model
    python bootstrap.py --model 1.2B     # Download only 1.2B model
    python bootstrap.py --model 2.6B     # Download only 2.6B model
    python bootstrap.py --skip-models    # Skip model downloads
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model definitions
MODELS = {
    "700M": {
        "id": "LiquidAI/LFM2-700M",
        "size": "~742M parameters",
        "use_case": "Fast inference, validation, RAG runtime",
        "min_ram": "4GB",
        "disk_space": "~3GB",
    },
    "1.2B": {
        "id": "LiquidAI/LFM2-1.2B",
        "size": "~1.17B parameters",
        "use_case": "Balanced tasks, summarization, rewriting",
        "min_ram": "6GB",
        "disk_space": "~5GB",
    },
    "2.6B": {
        "id": "LiquidAI/LFM2-2.6B",
        "size": "~2.57B parameters",
        "use_case": "Quality-focused tasks, chunking, metadata, QA generation",
        "min_ram": "10GB",
        "disk_space": "~10GB",
    },
}


def check_huggingface_cli() -> bool:
    """Check if huggingface-hub CLI is available."""
    try:
        import huggingface_hub
        logger.info(f"✓ huggingface-hub {huggingface_hub.__version__} is installed")
        return True
    except ImportError:
        logger.error("✗ huggingface-hub is not installed")
        logger.info("Install it with: pip install huggingface-hub")
        return False


def get_models_dir() -> Path:
    """Get the models directory path."""
    # Check if we're in the project root
    project_root = Path(__file__).parent
    models_dir = project_root / "models"

    if not models_dir.exists():
        logger.warning(f"Models directory does not exist: {models_dir}")
        logger.info("Creating models directory...")
        models_dir.mkdir(parents=True, exist_ok=True)

    return models_dir


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if sufficient disk space is available."""
    try:
        import shutil
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024**3)

        if available_gb < required_gb:
            logger.warning(
                f"Low disk space: {available_gb:.1f}GB available, "
                f"{required_gb:.1f}GB recommended"
            )
            return False

        logger.info(f"✓ Disk space: {available_gb:.1f}GB available")
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Continue anyway


def model_exists(model_path: Path) -> bool:
    """Check if a model has already been downloaded."""
    if not model_path.exists():
        return False

    # Check for essential model files
    required_files = ["config.json"]
    has_weights = any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin"))

    has_all = all((model_path / f).exists() for f in required_files) and has_weights

    if has_all:
        logger.info(f"✓ Model already exists: {model_path}")
        return True

    return False


def download_model(model_id: str, model_name: str, models_dir: Path, force: bool = False) -> bool:
    """
    Download a LiquidAI model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., "LiquidAI/LFM2-700M")
        model_name: Model name for local directory (e.g., "LFM2-700M")
        models_dir: Directory to save models
        force: Force re-download even if model exists

    Returns:
        True if successful, False otherwise
    """
    from huggingface_hub import snapshot_download

    model_path = models_dir / model_name

    # Check if already downloaded
    if not force and model_exists(model_path):
        logger.info(f"Skipping {model_name} (already downloaded)")
        return True

    logger.info(f"Downloading {model_id} to {model_path}...")
    logger.info(f"  Size: {MODELS[model_name.split('-')[-1]]['size']}")
    logger.info(f"  Use case: {MODELS[model_name.split('-')[-1]]['use_case']}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            # Download only model weights and config, not the full repo
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.model",
                "tokenizer.json",
                "*.txt",
            ],
            ignore_patterns=[
                "*.msgpack",
                "*.h5",
                "*.ot",
            ],
        )
        logger.info(f"✓ Successfully downloaded {model_name}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False


def setup_environment_info() -> None:
    """Display environment setup information."""
    logger.info("=" * 60)
    logger.info("Environment Setup Information")
    logger.info("=" * 60)

    # Python version
    logger.info(f"Python: {sys.version.split()[0]}")

    # Check key dependencies
    deps_status = []
    for dep in ["torch", "transformers", "pydantic", "chromadb", "fastapi"]:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            deps_status.append(f"✓ {dep} {version}")
        except ImportError:
            deps_status.append(f"✗ {dep} (not installed)")

    for status in deps_status:
        logger.info(status)

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.version.cuda}")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"  VRAM: {props.total_memory / (1024**3):.1f}GB")
        else:
            logger.info("✗ CUDA not available (CPU-only mode)")
    except ImportError:
        logger.info("✗ torch not installed (cannot check CUDA)")

    logger.info("=" * 60)


def list_models() -> None:
    """List available models and their information."""
    logger.info("=" * 60)
    logger.info("Available LiquidAI Models")
    logger.info("=" * 60)

    for model_key, info in MODELS.items():
        logger.info(f"\n{model_key}:")
        logger.info(f"  ID: {info['id']}")
        logger.info(f"  Size: {info['size']}")
        logger.info(f"  Use case: {info['use_case']}")
        logger.info(f"  Min RAM: {info['min_ram']}")
        logger.info(f"  Disk space: {info['disk_space']}")

    logger.info("=" * 60)


def main() -> int:
    """Main bootstrap function."""
    parser = argparse.ArgumentParser(
        description="Bootstrap LiquidAI Stack - Download models and setup environment"
    )
    parser.add_argument(
        "--model",
        choices=["700M", "1.2B", "2.6B", "all"],
        default="all",
        help="Which model(s) to download (default: all)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model downloads, only check environment",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models exist",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Custom models directory (default: ./models)",
    )

    args = parser.parse_args()

    # List models and exit
    if args.list:
        list_models()
        return 0

    # Display environment info
    setup_environment_info()

    # Skip model downloads if requested
    if args.skip_models:
        logger.info("Skipping model downloads (--skip-models)")
        return 0

    # Check for huggingface-hub
    if not check_huggingface_cli():
        logger.error("Please install huggingface-hub first:")
        logger.error("  pip install huggingface-hub")
        return 1

    # Get models directory
    models_dir = args.models_dir or get_models_dir()
    logger.info(f"Models directory: {models_dir}")

    # Check disk space (rough estimate)
    required_space = {
        "700M": 3,
        "1.2B": 5,
        "2.6B": 10,
        "all": 18,
    }
    check_disk_space(models_dir, required_space.get(args.model, 20))

    # Determine which models to download
    if args.model == "all":
        models_to_download = ["700M", "1.2B", "2.6B"]
    else:
        models_to_download = [args.model]

    # Download models
    logger.info("=" * 60)
    logger.info("Downloading Models")
    logger.info("=" * 60)

    success_count = 0
    for model_key in models_to_download:
        model_info = MODELS[model_key]
        model_id = model_info["id"]
        model_name = model_id.split("/")[1]  # Extract "LFM2-700M" from "LiquidAI/LFM2-700M"

        if download_model(model_id, model_name, models_dir, force=args.force):
            success_count += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Bootstrap Summary")
    logger.info("=" * 60)
    logger.info(f"Downloaded: {success_count}/{len(models_to_download)} models")

    if success_count == len(models_to_download):
        logger.info("✓ All models downloaded successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Place documents in: data/raw/")
        logger.info("  2. Run ETL pipeline: uv run liquid-etl")
        logger.info("  3. Start RAG server: uv run liquid-rag-server --port 8000")
        return 0
    else:
        logger.warning("⚠ Some models failed to download")
        return 1


if __name__ == "__main__":
    sys.exit(main())
