"""Generation quality evaluation metrics for fine-tuning data assessment.

Implements:
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- BLEU (Bilingual Evaluation Understudy)
- Perplexity
- Exact Match
- F1 Score
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies for advanced metrics
try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available. Install with: uv add rouge-score")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class GenerationMetrics:
    """Container for generation evaluation metrics."""

    # Token-level metrics
    rouge1_f: float = 0.0  # ROUGE-1 F1
    rouge1_p: float = 0.0  # ROUGE-1 Precision
    rouge1_r: float = 0.0  # ROUGE-1 Recall
    rouge2_f: float = 0.0  # ROUGE-2 F1
    rougeL_f: float = 0.0  # ROUGE-L F1

    bleu: float = 0.0  # BLEU score

    # Sequence-level metrics
    exact_match: float = 0.0  # Exact match rate
    token_f1: float = 0.0  # Token-level F1

    # Model-based metrics
    perplexity: float | None = None

    # Statistics
    num_samples: int = 0
    avg_prediction_length: float = 0.0
    avg_reference_length: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "rouge1_f": self.rouge1_f,
            "rouge1_p": self.rouge1_p,
            "rouge1_r": self.rouge1_r,
            "rouge2_f": self.rouge2_f,
            "rougeL_f": self.rougeL_f,
            "bleu": self.bleu,
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "perplexity": self.perplexity,
            "num_samples": self.num_samples,
            "avg_prediction_length": self.avg_prediction_length,
            "avg_reference_length": self.avg_reference_length,
        }

    def __str__(self) -> str:
        """Format metrics as string."""
        lines = [
            f"Generation Metrics (n={self.num_samples} samples)",
            f"  ROUGE-1 F1: {self.rouge1_f:.4f}",
            f"  ROUGE-2 F1: {self.rouge2_f:.4f}",
            f"  ROUGE-L F1: {self.rougeL_f:.4f}",
            f"  BLEU: {self.bleu:.4f}",
            f"  Exact Match: {self.exact_match:.4f}",
            f"  Token F1: {self.token_f1:.4f}",
        ]

        if self.perplexity is not None:
            lines.append(f"  Perplexity: {self.perplexity:.4f}")

        return "\n".join(lines)


class GenerationEvaluator:
    """
    Evaluator for generation quality.

    Computes ROUGE, BLEU, and other metrics for evaluating generated text
    against references (e.g., for fine-tuning data quality assessment).
    """

    def __init__(
        self,
        compute_rouge: bool = True,
        compute_bleu: bool = True,
        compute_perplexity: bool = False,
        perplexity_model: str | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            compute_rouge: Whether to compute ROUGE scores
            compute_bleu: Whether to compute BLEU scores
            compute_perplexity: Whether to compute perplexity
            perplexity_model: Model ID for perplexity computation
        """
        self.compute_rouge = compute_rouge and ROUGE_AVAILABLE
        self.compute_bleu = compute_bleu
        self.compute_perplexity = compute_perplexity and TRANSFORMERS_AVAILABLE

        if compute_rouge and not ROUGE_AVAILABLE:
            logger.warning("ROUGE requested but rouge-score not available")

        # Initialize ROUGE scorer
        self.rouge_scorer = None
        if self.compute_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=True,
            )

        # Initialize perplexity model if requested
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        if self.compute_perplexity and perplexity_model:
            self._load_perplexity_model(perplexity_model)

    def _load_perplexity_model(self, model_id: str) -> None:
        """Load model for perplexity computation."""
        try:
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.perplexity_model.eval()
            logger.info(f"Loaded perplexity model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load perplexity model: {e}")
            self.compute_perplexity = False

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.

        Args:
            predictions: List of predicted/generated texts
            references: List of reference texts

        Returns:
            GenerationMetrics with computed scores
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        metrics = GenerationMetrics(num_samples=len(predictions))

        # Compute ROUGE
        if self.compute_rouge:
            rouge_scores = self._compute_rouge(predictions, references)
            metrics.rouge1_f = rouge_scores["rouge1_f"]
            metrics.rouge1_p = rouge_scores["rouge1_p"]
            metrics.rouge1_r = rouge_scores["rouge1_r"]
            metrics.rouge2_f = rouge_scores["rouge2_f"]
            metrics.rougeL_f = rouge_scores["rougeL_f"]

        # Compute BLEU
        if self.compute_bleu:
            metrics.bleu = self._compute_bleu(predictions, references)

        # Compute exact match and token F1
        exact_matches = []
        token_f1_scores = []

        for pred, ref in zip(predictions, references):
            # Normalize for comparison
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)

            # Exact match
            exact_matches.append(float(pred_norm == ref_norm))

            # Token F1
            token_f1_scores.append(self._token_f1(pred_norm, ref_norm))

        metrics.exact_match = float(np.mean(exact_matches))
        metrics.token_f1 = float(np.mean(token_f1_scores))

        # Compute lengths
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        metrics.avg_prediction_length = float(np.mean(pred_lengths))
        metrics.avg_reference_length = float(np.mean(ref_lengths))

        # Compute perplexity if requested
        if self.compute_perplexity and self.perplexity_model:
            metrics.perplexity = self._compute_perplexity(predictions)

        logger.info(f"Evaluated {metrics.num_samples} samples")
        return metrics

    def _compute_rouge(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        """Compute ROUGE scores."""
        if not self.rouge_scorer:
            return {}

        rouge1_f_scores = []
        rouge1_p_scores = []
        rouge1_r_scores = []
        rouge2_f_scores = []
        rougeL_f_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)

            rouge1_f_scores.append(scores["rouge1"].fmeasure)
            rouge1_p_scores.append(scores["rouge1"].precision)
            rouge1_r_scores.append(scores["rouge1"].recall)
            rouge2_f_scores.append(scores["rouge2"].fmeasure)
            rougeL_f_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1_f": float(np.mean(rouge1_f_scores)),
            "rouge1_p": float(np.mean(rouge1_p_scores)),
            "rouge1_r": float(np.mean(rouge1_r_scores)),
            "rouge2_f": float(np.mean(rouge2_f_scores)),
            "rougeL_f": float(np.mean(rougeL_f_scores)),
        }

    def _compute_bleu(
        self, predictions: list[str], references: list[str]
    ) -> float:
        """Compute corpus-level BLEU score."""
        bleu_scores = []

        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            # Compute sentence-level BLEU
            score = self._sentence_bleu(pred_tokens, [ref_tokens])
            bleu_scores.append(score)

        return float(np.mean(bleu_scores))

    @staticmethod
    def _sentence_bleu(
        candidate: list[str],
        references: list[list[str]],
        max_n: int = 4,
    ) -> float:
        """Compute sentence-level BLEU score."""
        # Compute n-gram precisions
        precisions = []

        for n in range(1, max_n + 1):
            # Count n-grams in candidate
            candidate_ngrams = Counter()
            for i in range(len(candidate) - n + 1):
                ngram = tuple(candidate[i : i + n])
                candidate_ngrams[ngram] += 1

            # Count max n-grams in references
            max_ref_ngrams: Counter = Counter()
            for reference in references:
                ref_ngrams = Counter()
                for i in range(len(reference) - n + 1):
                    ngram = tuple(reference[i : i + n])
                    ref_ngrams[ngram] += 1
                for ngram in ref_ngrams:
                    max_ref_ngrams[ngram] = max(
                        max_ref_ngrams[ngram], ref_ngrams[ngram]
                    )

            # Compute clipped counts
            clipped_counts = sum(
                min(count, max_ref_ngrams[ngram])
                for ngram, count in candidate_ngrams.items()
            )
            total_counts = max(1, len(candidate) - n + 1)

            precision = clipped_counts / total_counts if total_counts > 0 else 0
            precisions.append(precision)

        # Geometric mean of precisions
        if min(precisions) == 0:
            return 0.0

        geo_mean = np.exp(np.mean(np.log(precisions)))

        # Brevity penalty
        ref_length = len(references[0]) if references else 0
        candidate_length = len(candidate)

        if candidate_length >= ref_length:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_length / candidate_length) if candidate_length > 0 else 0

        return float(bp * geo_mean)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def _token_f1(prediction: str, reference: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(prediction.split())
        ref_tokens = set(reference.split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = pred_tokens & ref_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)

    def _compute_perplexity(self, texts: list[str]) -> float:
        """Compute average perplexity across texts."""
        if not self.perplexity_model or not self.perplexity_tokenizer:
            return 0.0

        perplexities = []

        with torch.no_grad():
            for text in texts:
                try:
                    # Tokenize
                    inputs = self.perplexity_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(self.perplexity_model.device) for k, v in inputs.items()}

                    # Get loss
                    outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()

                    # Perplexity = exp(loss)
                    ppl = np.exp(loss)
                    perplexities.append(ppl)

                except Exception as e:
                    logger.warning(f"Failed to compute perplexity for text: {e}")

        if not perplexities:
            return 0.0

        return float(np.mean(perplexities))


def evaluate_generation_from_dict(
    evaluation_data: dict[str, Any],
    compute_rouge: bool = True,
    compute_bleu: bool = True,
) -> GenerationMetrics:
    """
    Convenience function to evaluate generation from dictionary format.

    Expected format:
    {
        "predictions": ["pred1", "pred2", ...],
        "references": ["ref1", "ref2", ...]
    }

    Args:
        evaluation_data: Dictionary with predictions and references
        compute_rouge: Whether to compute ROUGE
        compute_bleu: Whether to compute BLEU

    Returns:
        GenerationMetrics with computed scores
    """
    evaluator = GenerationEvaluator(
        compute_rouge=compute_rouge,
        compute_bleu=compute_bleu,
    )

    predictions = evaluation_data["predictions"]
    references = evaluation_data["references"]

    return evaluator.evaluate(predictions, references)
