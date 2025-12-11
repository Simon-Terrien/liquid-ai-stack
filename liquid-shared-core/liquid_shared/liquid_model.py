# liquid_shared/liquid_model.py
"""
Local LiquidAI model wrapper compatible with Pydantic AI.

Uses the OutlinesModel from Pydantic AI for local transformers inference,
with automatic device selection (CPU/GPU) based on available hardware.
"""
import logging

import torch
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE
from .devices import DeviceConfig, recommend_device

logger = logging.getLogger(__name__)


class LocalLiquidModel:
    """
    Local LiquidAI model wrapper for Pydantic AI.
    
    Uses OutlinesModel.from_transformers() for integration with Pydantic AI,
    enabling structured output generation and tool calling.
    
    Example:
        ```python
        from liquid_shared.liquid_model import LocalLiquidModel
        from pydantic_ai import Agent
        
        model = LocalLiquidModel("LiquidAI/LFM2-1.2B")
        agent = Agent(model.get_pydantic_model())
        result = agent.run_sync("Hello!")
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        device_config: DeviceConfig | None = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize a local LiquidAI model.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            max_new_tokens: Default max tokens for generation
            temperature: Default temperature for generation
            device_config: Override automatic device selection
            trust_remote_code: Whether to trust remote code (default False for security)
        """
        self.model_name = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Get device configuration
        if device_config is None:
            device_config = recommend_device(model_name_or_path)

        self.device_config = device_config
        self.device = torch.device(device_config.device)
        self.dtype = device_config.torch_dtype

        logger.info(f"Loading model {model_name_or_path}")
        logger.info(f"Device config: {device_config.reason}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        # Load model with appropriate settings
        model_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": trust_remote_code,
        }

        if device_config.device == "cuda":
            # Use explicit device mapping to avoid meta device issues
            # Place entire model on the specified CUDA device
            model_kwargs["device_map"] = {"": "cuda:0"}
        else:
            # For CPU, load without device_map and manually move
            model_kwargs["low_cpu_mem_usage"] = True

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )

        if device_config.device == "cpu":
            self._hf_model.to(self.device)

        self._hf_model.eval()

        # Create Pydantic AI compatible model
        self._pydantic_model = OutlinesModel.from_transformers(
            self._hf_model,
            self.tokenizer
        )

        logger.info(f"Model loaded successfully on {device_config.device}")

    def get_pydantic_model(self) -> OutlinesModel:
        """Get the Pydantic AI compatible model instance."""
        return self._pydantic_model

    def get_default_settings(self) -> ModelSettings:
        """Get default model settings for generation."""
        return ModelSettings(
            extra_body={
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Direct generation without Pydantic AI wrapper.
        
        Useful for simple text generation tasks.
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else None,
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "reason": self.device_config.reason,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    def cleanup(self) -> None:
        """Clean up model from memory."""
        import gc

        if hasattr(self, '_hf_model'):
            del self._hf_model
        if hasattr(self, '_pydantic_model'):
            del self._pydantic_model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info(f"Cleaned up model {self.model_name} from memory")


# Pre-configured model factories for common use cases

def create_quality_model(model_path: str = "LiquidAI/LFM2-2.6B") -> LocalLiquidModel:
    """Create a quality-focused model (2.6B) for ETL tasks."""
    return LocalLiquidModel(
        model_path,
        max_new_tokens=1024,
        temperature=0.1,  # Lower temp for more deterministic output
    )


def create_balanced_model(model_path: str = "LiquidAI/LFM2-1.2B") -> LocalLiquidModel:
    """Create a balanced model (1.2B) for summarization and rewriting."""
    return LocalLiquidModel(
        model_path,
        max_new_tokens=512,
        temperature=0.2,
    )


def create_fast_model(model_path: str = "LiquidAI/LFM2-700M") -> LocalLiquidModel:
    """Create a fast model (700M) for RAG inference and validation."""
    return LocalLiquidModel(
        model_path,
        max_new_tokens=256,
        temperature=0.2,
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing LocalLiquidModel...")

    # Try to load the smallest model
    try:
        model = create_fast_model()
        print(f"Model info: {model.info}")

        # Simple generation test
        response = model.generate("Hello, I am a", max_new_tokens=20)
        print(f"Generated: {response}")
    except Exception as e:
        print(f"Could not load model (expected if not downloaded): {e}")
