from __future__ import annotations

import argparse
import gc
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_sequences import get_refusal_sequences, normalize_language


SAFE_PROMPT_COLUMN_CANDIDATES = [
    "Safe_Control_Prompt",
    "safe_control_prompt",
    "Safe Prompt",
    "safe_prompt",
    "prompt",
    "Prompt",
    "inputs",
    "text",
]

SAFE_LANG_COLUMN_CANDIDATES = [
    "inputs",
    "input",
    "safe_prompt",
    "Safe_Prompt",
    "Safe Prompt",
    "prompt",
    "Prompt",
    "text",
    "Text",
    "Safe_Control_Prompt",
    "label",
    "labels"
]

ENGLISH_QUESTION_COLUMN_CANDIDATES = [
    "English Question",
    "English_Question",
    "english_question",
    "English prompt",
    "english_prompt",
]

LITERAL_COLUMN_CANDIDATES = [
    "Input_A_Literal",
    "Input A Literal",
    "literal",
    "Literal",
]

METAPHOR_COLUMN_CANDIDATES = [
    "Input_B_Metaphor",
    "Input B Metaphor",
    "metaphor",
    "Metaphor",
]

HARM_TYPE_COLUMN_CANDIDATES = [
    "types_of_harm",
    "Types_of_Harm",
    "type_of_harm",
    "harm_type",
    "Harm_Type",
]

DEFAULT_HIDDEN_BATCH_SIZE = 4
DEFAULT_GENERATION_BATCH_SIZE = 2
DEFAULT_SEQUENCE_CLL_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_PROBE_EPOCHS = 200
DEFAULT_PROBE_LR = 1e-2
DEFAULT_PROBE_WEIGHT_DECAY = 1e-4
DEFAULT_PROBE_VAL_FRACTION = 0.2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ModelPreset:
    model_id: str
    output_name: str
    total_layers: int
    layer_start: int
    layer_end: int
    pca_layer: int

    @property
    def target_layers(self) -> Tuple[int, ...]:
        return tuple(range(self.layer_start, self.layer_end + 1))


MODEL_PRESETS: Dict[str, ModelPreset] = {
        "mistralai/Mistral-7B-Instruct-v0.3": ModelPreset(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        output_name="mistral",
        total_layers=32,
        layer_start=14,
        layer_end=24,
        pca_layer=16,
    ),    "Qwen/Qwen3-8B": ModelPreset(
        model_id="Qwen/Qwen3-8B",
        output_name="qwen",
        total_layers=36,
        layer_start=16,
        layer_end=28,
        pca_layer=20,
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelPreset(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        output_name="llama",
        total_layers=32,
        layer_start=14,
        layer_end=24,
        pca_layer=16,
    ),
}

MODEL_ALIASES = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen3-8B",
    "qwen3": "Qwen/Qwen3-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
}

_PLOT_MODULES: Optional[Tuple[object, object]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multilingual refusal-vector, probe, generation, and PCA analyses "
            "for literal vs metaphor prompts across supported instruction-tuned models."
        )
    )
    parser.add_argument("--language", required=True, help="Language code or name, e.g. amh, twi, hausa.")
    parser.add_argument(
        "--model-id",
        required=True,
        help=(
            "Model ID or alias. Supported presets: mistralai/Mistral-7B-Instruct-v0.3, "
            "Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct, or aliases mistral/qwen/llama."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Short model name to use in output folder/file names. Defaults to preset alias or sanitized model ID.",
    )
    parser.add_argument("--experiment-csv", required=True, help="Path to {lang}_ready_for_experiment.csv")
    parser.add_argument("--safe-lang-csv", required=True, help="Path to {lang}_safe.csv")
    parser.add_argument("--safe-eng-csv", required=True, help="Path to Safe_prompts_eng.csv")
    parser.add_argument("--output-root", default=".", help="Root directory where the output folder will be created.")
    parser.add_argument("--english-column", default=None, help="Override the unsafe English question column.")
    parser.add_argument("--literal-column", default=None, help="Override the Input_A_Literal column.")
    parser.add_argument("--metaphor-column", default=None, help="Override the Input_B_Metaphor column.")
    parser.add_argument("--safe-eng-column", default=None, help="Override the safe English prompt column.")
    parser.add_argument("--safe-lang-column", default=None, help="Override the safe language prompt column.")
    parser.add_argument("--harm-type-column", default=None, help="Override the harm-type column for grouped plots.")
    parser.add_argument("--layer-start", type=int, default=None, help="Optional custom layer-range start.")
    parser.add_argument("--layer-end", type=int, default=None, help="Optional custom layer-range end.")
    parser.add_argument("--pca-layer", type=int, default=None, help="Optional custom PCA layer.")
    parser.add_argument("--hidden-batch-size", type=int, default=DEFAULT_HIDDEN_BATCH_SIZE)
    parser.add_argument("--generation-batch-size", type=int, default=DEFAULT_GENERATION_BATCH_SIZE)
    parser.add_argument("--sequence-cll-batch-size", type=int, default=DEFAULT_SEQUENCE_CLL_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--probe-epochs", type=int, default=DEFAULT_PROBE_EPOCHS)
    parser.add_argument("--probe-lr", type=float, default=DEFAULT_PROBE_LR)
    parser.add_argument("--probe-weight-decay", type=float, default=DEFAULT_PROBE_WEIGHT_DECAY)
    parser.add_argument("--probe-val-fraction", type=float, default=DEFAULT_PROBE_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. If omitted, HF_TOKEN or HUGGINGFACE_TOKEN will be used.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to AutoModelForCausalLM.from_pretrained. Default: auto",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when required by a model/tokenizer.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    return cleaned.strip("_") or "artifact"


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose_torch_dtype() -> torch.dtype:
    return torch.float16


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


def resolve_model_preset(
    requested_model_id: str,
    requested_model_name: Optional[str],
    layer_start: Optional[int],
    layer_end: Optional[int],
    pca_layer: Optional[int],
) -> Tuple[str, str, Tuple[int, ...], int, Optional[ModelPreset]]:
    resolved_model_id = MODEL_ALIASES.get(requested_model_id, requested_model_id)
    preset = MODEL_PRESETS.get(resolved_model_id)

    if preset is not None and layer_start is None and layer_end is None and pca_layer is None:
        model_name = sanitize_name(requested_model_name or preset.output_name)
        return preset.model_id, model_name, preset.target_layers, preset.pca_layer, preset

    if layer_start is None or layer_end is None or pca_layer is None:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(
            "Unsupported model without full custom layer settings. "
            f"Provide --layer-start, --layer-end, and --pca-layer, or use one of: {supported}"
        )

    if layer_start > layer_end:
        raise ValueError("--layer-start must be less than or equal to --layer-end")

    model_name = sanitize_name(requested_model_name or resolved_model_id.split("/")[-1])
    return resolved_model_id, model_name, tuple(range(layer_start, layer_end + 1)), pca_layer, preset


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for device in hf_device_map.values():
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
            if isinstance(device, str) and device == "cpu":
                return torch.device("cpu")
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
    return next(model.parameters()).device


def format_chat_prompts(tokenizer: AutoTokenizer, prompts: Sequence[str]) -> List[str]:
    formatted: List[str] = []
    for prompt in prompts:
        if getattr(tokenizer, "chat_template", None):
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(prompt)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            formatted.append(str(prompt))
    return formatted


def get_last_token_hidden_states(
    prompts: Sequence[str],
    layers: Sequence[int],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    batch_size: int,
) -> Dict[int, torch.Tensor]:
    if not prompts:
        raise ValueError("No prompts were provided for hidden-state extraction.")

    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    all_layer_states = {layer: [] for layer in layers}

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        formatted_prompts = format_chat_prompts(tokenizer, batch_prompts)
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(input_device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        last_indices = inputs["attention_mask"].sum(dim=1) - 1

        for layer in layers:
            hidden = outputs.hidden_states[layer]
            batch_positions = torch.arange(hidden.size(0), device=hidden.device)
            token_positions = last_indices.to(hidden.device)
            batch_states = hidden[batch_positions, token_positions, :].detach().float().cpu()
            all_layer_states[layer].append(batch_states)

        del outputs, inputs
        clear_memory()

    tokenizer.padding_side = "right"
    return {
        layer: torch.cat(layer_batches, dim=0)
        for layer, layer_batches in all_layer_states.items()
    }


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.int()
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return float("nan")

    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    true_positives = torch.cumsum((sorted_labels == 1).float(), dim=0)
    false_positives = torch.cumsum((sorted_labels == 0).float(), dim=0)

    tpr = true_positives / positives
    fpr = false_positives / negatives

    tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
    fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])

    return float(torch.trapz(tpr, fpr).item())


def validation_size(count: int, fraction: float) -> int:
    if count < 2:
        raise ValueError("Linear probe training requires at least two examples per class.")
    proposed = max(1, int(count * fraction))
    return min(proposed, count - 1)


def train_linear_probe(
    safe_states: torch.Tensor,
    unsafe_states: torch.Tensor,
    layer: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    if safe_states.ndim != 2 or unsafe_states.ndim != 2:
        raise ValueError("Probe training expects 2D hidden-state tensors.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer)

    labels_safe = torch.zeros(len(safe_states), dtype=torch.float32)
    labels_unsafe = torch.ones(len(unsafe_states), dtype=torch.float32)

    X = torch.cat([safe_states, unsafe_states], dim=0).float()
    y = torch.cat([labels_safe, labels_unsafe], dim=0)

    safe_perm = torch.randperm(len(safe_states), generator=generator)
    unsafe_perm = torch.randperm(len(unsafe_states), generator=generator)

    safe_val_size = validation_size(len(safe_states), val_fraction)
    unsafe_val_size = validation_size(len(unsafe_states), val_fraction)

    safe_val_idx = safe_perm[:safe_val_size]
    safe_train_idx = safe_perm[safe_val_size:]
    unsafe_val_idx = unsafe_perm[:unsafe_val_size] + len(safe_states)
    unsafe_train_idx = unsafe_perm[unsafe_val_size:] + len(safe_states)

    train_idx = torch.cat([safe_train_idx, unsafe_train_idx], dim=0)
    val_idx = torch.cat([safe_val_idx, unsafe_val_idx], dim=0)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    feature_mean = X_train.mean(dim=0, keepdim=True)
    feature_std = X_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std

    probe = nn.Linear(X_train.size(1), 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    positive_count = max(float((y_train == 1).sum().item()), 1.0)
    negative_count = max(float((y_train == 0).sum().item()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative_count / positive_count))

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        train_logits = probe(X_train).squeeze(-1)
        loss = loss_fn(train_logits, y_train)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        train_logits = probe(X_train).squeeze(-1)
        val_logits = probe(X_val).squeeze(-1)

        train_probs = torch.sigmoid(train_logits)
        val_probs = torch.sigmoid(val_logits)

        train_preds = (train_probs >= 0.5).float()
        val_preds = (val_probs >= 0.5).float()

    bundle = {
        "weight": probe.weight.detach().clone().cpu(),
        "bias": probe.bias.detach().clone().cpu(),
        "feature_mean": feature_mean.detach().clone().cpu(),
        "feature_std": feature_std.detach().clone().cpu(),
    }

    metrics = {
        "layer": float(layer),
        "train_accuracy": float((train_preds == y_train).float().mean().item()),
        "val_accuracy": float((val_preds == y_val).float().mean().item()),
        "train_auroc": binary_auroc(train_probs, y_train),
        "val_auroc": binary_auroc(val_probs, y_val),
        "safe_train_count": float(len(safe_train_idx)),
        "unsafe_train_count": float(len(unsafe_train_idx)),
        "safe_val_count": float(len(safe_val_idx)),
        "unsafe_val_count": float(len(unsafe_val_idx)),
    }

    del probe, X, y, X_train, X_val, y_train, y_val
    clear_memory()
    return bundle, metrics


def apply_linear_probe(hidden_states: torch.Tensor, probe_bundle: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    normalized = (hidden_states.float() - probe_bundle["feature_mean"]) / probe_bundle["feature_std"]
    logits = normalized @ probe_bundle["weight"].T + probe_bundle["bias"]
    logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits)
    return logits, probs


def tokenize_candidate_sequences(tokenizer: AutoTokenizer, sequences: Sequence[str]) -> List[Tuple[str, List[int]]]:
    tokenized: List[Tuple[str, List[int]]] = []
    seen = set()
    for text in sequences:
        if text in seen:
            continue
        seen.add(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            tokenized.append((text, token_ids))
    return tokenized


def get_sequence_level_refusal_metrics(
    prompt: str,
    candidate_sequences: Sequence[Tuple[str, List[int]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"

    prompt_text = format_chat_prompts(tokenizer, [prompt])[0]
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")

    all_candidate_scores = []

    for start in range(0, len(candidate_sequences), batch_size):
        batch = candidate_sequences[start : start + batch_size]
        full_sequences = [prompt_ids + continuation_ids for _, continuation_ids in batch]

        model_inputs = tokenizer.pad(
            {
                "input_ids": full_sequences,
                "attention_mask": [[1] * len(ids) for ids in full_sequences],
            },
            return_tensors="pt",
            padding=True,
        ).to(input_device)

        with torch.no_grad():
            outputs = model(**model_inputs, return_dict=True)
            log_probs = F.log_softmax(outputs.logits, dim=-1)

        for batch_index, (text, continuation_ids) in enumerate(batch):
            total_logprob = 0.0
            for token_offset, token_id in enumerate(continuation_ids):
                position = len(prompt_ids) - 1 + token_offset
                total_logprob += float(log_probs[batch_index, position, token_id].item())

            avg_logprob = total_logprob / len(continuation_ids)
            all_candidate_scores.append(
                {
                    "text": text,
                    "total_logprob": total_logprob,
                    "avg_logprob": avg_logprob,
                    "token_length": float(len(continuation_ids)),
                }
            )

        del outputs, log_probs, model_inputs
        clear_memory()

    best_by_avg = max(all_candidate_scores, key=lambda item: item["avg_logprob"])
    best_by_total = max(all_candidate_scores, key=lambda item: item["total_logprob"])
    approx_any_logprob = float(
        torch.logsumexp(
            torch.tensor([item["total_logprob"] for item in all_candidate_scores], dtype=torch.float32),
            dim=0,
        ).item()
    )

    return {
        "best_refusal_sequence_avg": best_by_avg["text"],
        "best_refusal_sequence_total": best_by_total["text"],
        "seq_cll_best_avg": best_by_avg["avg_logprob"],
        "seq_cll_best_total": best_by_total["total_logprob"],
        "seq_cll_any_refusal_logprob": approx_any_logprob,
    }


def generate_first_tokens_for_prompts(
    prompts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    max_new_tokens: int,
    batch_size: int,
) -> List[Dict[str, str]]:
    if not prompts:
        return []

    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"
    records: List[Dict[str, str]] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        formatted_prompts = format_chat_prompts(tokenizer, batch_prompts)
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(input_device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_width = inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_width:]

        for batch_index, prompt in enumerate(batch_prompts):
            token_ids = generated_only[batch_index][:max_new_tokens].detach().cpu().tolist()
            token_strings = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]
            records.append(
                {
                    "prompt": prompt,
                    "token_ids": " ".join(map(str, token_ids)),
                    "token_texts": " | ".join(token_strings),
                    "decoded_text": tokenizer.decode(token_ids, skip_special_tokens=True),
                }
            )

        del generated, generated_only, inputs
        clear_memory()

    tokenizer.padding_side = "right"
    return records


def infer_column(df: pd.DataFrame, override: Optional[str], candidates: Sequence[str], label: str) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"{label} override column '{override}' was not found in the CSV.")
        return override

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(f"Unable to find a column for {label}. Available columns: {list(df.columns)}")


def maybe_infer_column(df: pd.DataFrame, override: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if override:
        if override not in df.columns:
            raise ValueError(f"Override column '{override}' was not found in the CSV.")
        return override

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def safe_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_product = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if norm_product == 0.0:
        return float("nan")
    return float(np.dot(vec_a, vec_b) / norm_product)


def artifact_path(output_dir: Path, language: str, model_name: str, description: str, suffix: str) -> Path:
    filename = f"{sanitize_name(language)}_{sanitize_name(model_name)}_{sanitize_name(description)}{suffix}"
    return output_dir / filename


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def get_plot_modules() -> Tuple[object, object]:
    global _PLOT_MODULES
    if _PLOT_MODULES is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        _PLOT_MODULES = (plt, sns)
    return _PLOT_MODULES


def set_plot_theme() -> None:
    _, sns = get_plot_modules()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def save_internal_alignment_plot(
    df: pd.DataFrame,
    layers: Sequence[int],
    output_path: Path,
    literal_column: str,
    metaphor_column: str,
) -> None:
    plt, _ = get_plot_modules()
    plt.figure(figsize=(10, 6))
    mean_dot_a = [df[f"L{layer}_Dot_A"].mean() for layer in layers]
    mean_dot_b = [df[f"L{layer}_Dot_B"].mean() for layer in layers]
    plt.plot(layers, mean_dot_a, marker="o", label=f"{literal_column} (Input A)", color="navy", linewidth=2)
    plt.plot(layers, mean_dot_b, marker="s", label=f"{metaphor_column} (Input B)", color="darkred", linewidth=2)
    plt.title("Internal Alignment with Refusal Direction Across Layers", fontsize=14, pad=15)
    plt.xlabel("Hidden Layer", fontsize=12)
    plt.ylabel("Average Dot Product with Refusal Direction", fontsize=12)
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dot_drift_bar_plot(df: pd.DataFrame, layers: Sequence[int], output_path: Path) -> None:
    plt, sns = get_plot_modules()
    plt.figure(figsize=(10, 6))
    mean_drift = [df[f"L{layer}_Dot_Drift"].mean() for layer in layers]
    sns.barplot(x=list(layers), y=mean_drift, color="teal")
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)
    plt.title("Mean Drift Difference by Layer (Literal - Metaphor)", fontsize=14, pad=15)
    plt.xlabel("Hidden Layer", fontsize=12)
    plt.ylabel("Mean Dot-Product Drift", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_sequence_cll_scatter(df: pd.DataFrame, output_path: Path) -> None:
    plt, sns = get_plot_modules()
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x="A_seq_cll_best_avg",
        y="B_seq_cll_best_avg",
        alpha=0.7,
        color="purple",
        s=60,
    )
    min_val = min(df["A_seq_cll_best_avg"].min(), df["B_seq_cll_best_avg"].min())
    max_val = max(df["A_seq_cll_best_avg"].max(), df["B_seq_cll_best_avg"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="No Difference (y=x)")
    plt.title("Sequence-Level Refusal CLL: Literal vs Metaphor", fontsize=14, pad=15)
    plt.xlabel("Best Refusal Sequence Avg Log-Prob (Literal)", fontsize=12)
    plt.ylabel("Best Refusal Sequence Avg Log-Prob (Metaphor)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_probe_confidence_plot(df: pd.DataFrame, layers: Sequence[int], output_path: Path) -> None:
    plt, _ = get_plot_modules()
    plt.figure(figsize=(10, 6))
    mean_probe_a = [df[f"L{layer}_Probe_Prob_A"].mean() for layer in layers]
    mean_probe_b = [df[f"L{layer}_Probe_Prob_B"].mean() for layer in layers]
    plt.plot(layers, mean_probe_a, marker="o", label="Literal Translation (Input A)", color="navy", linewidth=2)
    plt.plot(layers, mean_probe_b, marker="s", label="Cultural Metaphor (Input B)", color="darkorange", linewidth=2)
    plt.title("Layer-wise Linear Probe Confidence Across Layers", fontsize=14, pad=15)
    plt.xlabel("Hidden Layer", fontsize=12)
    plt.ylabel("Average Probe Probability", fontsize=12)
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_probe_validation_plot(probe_df: pd.DataFrame, layers: Sequence[int], output_path: Path) -> None:
    plt, _ = get_plot_modules()
    probe_plot = probe_df.copy()
    probe_plot["layer"] = probe_plot["layer"].astype(int)
    plt.figure(figsize=(9, 5))
    plt.plot(probe_plot["layer"], probe_plot["val_accuracy"], marker="o", label="Validation Accuracy", linewidth=2)
    plt.plot(probe_plot["layer"], probe_plot["val_auroc"], marker="s", label="Validation AUROC", linewidth=2)
    plt.title("Linear Probe Validation Performance by Layer", fontsize=14, pad=15)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(layers)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_harm_category_plot(df: pd.DataFrame, harm_type_column: str, output_path: Path) -> None:
    plt, sns = get_plot_modules()
    harm_df = df.melt(
        id_vars=[harm_type_column],
        value_vars=["A_seq_cll_best_avg", "B_seq_cll_best_avg"],
        var_name="Input Type",
        value_name="Best Refusal Sequence Avg Log-Prob",
    )
    harm_df["Input Type"] = harm_df["Input Type"].replace(
        {
            "A_seq_cll_best_avg": "Literal Translation",
            "B_seq_cll_best_avg": "Cultural Metaphor",
        }
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=harm_df,
        x="Best Refusal Sequence Avg Log-Prob",
        y=harm_type_column,
        hue="Input Type",
        palette=["blue", "red"],
        errorbar=None,
    )
    plt.title("Refusal Strength by Harm Category", fontsize=14, pad=15)
    plt.xlabel("Average Best Refusal Sequence Log-Prob", fontsize=12)
    plt.ylabel("Harm Category", fontsize=12)
    plt.legend(title="Prompt Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pca_projection_plot(
    pca_safe_eng: np.ndarray,
    pca_unsafe_eng: np.ndarray,
    pca_safe_lang: np.ndarray,
    pca_unsafe_lang_lit: np.ndarray,
    pca_unsafe_lang_met: np.ndarray,
    output_path: Path,
    layer: int,
    language_label: str,
) -> None:
    plt, _ = get_plot_modules()
    plt.figure(figsize=(10, 8))

    def add_group(points: np.ndarray, label: str, color: str) -> None:
        plt.scatter(
            points[:, 0],
            points[:, 1],
            label=label,
            color=color,
            alpha=0.72,
            s=35,
            edgecolors="white",
            linewidths=0.4,
        )

    add_group(pca_safe_eng, "harmless_en", "#b0c4de")
    add_group(pca_unsafe_eng, "harmful_refuse_en", "#f4a460")
    add_group(pca_safe_lang, f"harmless_{language_label.lower()}", "#3cb371")
    add_group(pca_unsafe_lang_lit, f"harmful_refuse_{language_label.lower()}_lit", "#4682b4")
    add_group(pca_unsafe_lang_met, f"harmful_refuse_{language_label.lower()}_met", "#dc143c")

    mean_safe_eng_2d = np.mean(pca_safe_eng, axis=0)
    mean_unsafe_eng_2d = np.mean(pca_unsafe_eng, axis=0)
    mean_safe_lang_2d = np.mean(pca_safe_lang, axis=0)
    mean_unsafe_lang_lit_2d = np.mean(pca_unsafe_lang_lit, axis=0)
    mean_unsafe_lang_met_2d = np.mean(pca_unsafe_lang_met, axis=0)

    plt.annotate("", xy=mean_unsafe_eng_2d, xytext=mean_safe_eng_2d, arrowprops=dict(arrowstyle="->", color="black", lw=2.4))
    plt.annotate("", xy=mean_unsafe_lang_lit_2d, xytext=mean_safe_lang_2d, arrowprops=dict(arrowstyle="->", color="dimgray", lw=2.4))
    plt.annotate("", xy=mean_unsafe_lang_met_2d, xytext=mean_safe_lang_2d, arrowprops=dict(arrowstyle="->", color="crimson", lw=2.4))

    plt.title(f"PCA Projection on English Refusal Plane (Layer {layer})", fontsize=14, pad=15)
    plt.xlabel("Principal Component 1 (English Variance)", fontsize=12)
    plt.ylabel("Principal Component 2 (English Variance)", fontsize=12)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_pca_projection_plot_3d(
    pca_safe_eng: np.ndarray,
    pca_unsafe_eng: np.ndarray,
    pca_safe_lang: np.ndarray,
    pca_unsafe_lang_lit: np.ndarray,
    pca_unsafe_lang_met: np.ndarray,
    output_path: Path,
    layer: int,
    language_label: str,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Skipping 3D PCA plot.")
        return

    fig = go.Figure()

    def add_trace(points: np.ndarray, label: str, color: str) -> None:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            name=label,
            marker=dict(
                size=4,
                color=color,
                opacity=0.72,
                line=dict(width=0.5, color="white")
            )
        ))

    add_trace(pca_safe_eng, "harmless_en", "#b0c4de")
    add_trace(pca_unsafe_eng, "harmful_refuse_en", "#f4a460")
    add_trace(pca_safe_lang, f"harmless_{language_label.lower()}", "#3cb371")
    add_trace(pca_unsafe_lang_lit, f"harmful_refuse_{language_label.lower()}_lit", "#4682b4")
    add_trace(pca_unsafe_lang_met, f"harmful_refuse_{language_label.lower()}_met", "#dc143c")

    fig.update_layout(
        title=f"3D PCA Projection on English Refusal Space (Layer {layer})",
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1)
    )

    fig.write_html(str(output_path))

def save_cosine_similarity_plot(summary_df: pd.DataFrame, output_path: Path, layer: int) -> None:
    plt, sns = get_plot_modules()
    bar_df = summary_df[summary_df["metric"].str.contains("cosine_similarity")].copy()
    if bar_df.empty:
        return

    display_names = {
        "cosine_similarity_english_vs_literal": "English vs Literal",
        "cosine_similarity_english_vs_metaphor": "English vs Metaphor",
        "cosine_similarity_literal_vs_metaphor": "Literal vs Metaphor",
    }
    bar_df["label"] = bar_df["metric"].map(display_names).fillna(bar_df["metric"])

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=bar_df, x="label", y="value", palette=["steelblue", "crimson", "mediumpurple"][: len(bar_df)])
    plt.title(f"Cosine Similarity of Refusal Vectors (Layer {layer})", fontsize=14, pad=15)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.xlabel("")
    plt.ylim(-0.2, 1.0)
    for patch, value in zip(ax.patches, bar_df["value"]):
        ax.annotate(f"{value:.3f}", (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_probe_summary(probe_df: pd.DataFrame) -> None:
    print("\nLayer-wise linear probe summary")
    print("-" * 72)
    for _, row in probe_df.iterrows():
        print(
            f"Layer {int(row['layer'])}: "
            f"train_acc={row['train_accuracy']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, "
            f"train_auroc={row['train_auroc']:.4f}, "
            f"val_auroc={row['val_auroc']:.4f}"
        )
    print("-" * 72)


def align_generation_outputs(
    analysis_df: pd.DataFrame,
    literal_column: str,
    metaphor_column: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_device: torch.device,
    max_new_tokens: int,
    batch_size: int,
) -> pd.DataFrame:
    literal_prompts = analysis_df[literal_column].astype(str).tolist()
    metaphor_prompts = analysis_df[metaphor_column].astype(str).tolist()

    literal_records = generate_first_tokens_for_prompts(
        literal_prompts,
        tokenizer,
        model,
        input_device,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    metaphor_records = generate_first_tokens_for_prompts(
        metaphor_prompts,
        tokenizer,
        model,
        input_device,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    generation_df = analysis_df[["row_index", literal_column, metaphor_column]].copy()
    generation_df["Input_A_Generated_First_20_Token_IDs"] = [record["token_ids"] for record in literal_records]
    generation_df["Input_A_Generated_First_20_Token_Texts"] = [record["token_texts"] for record in literal_records]
    generation_df["Input_A_Generated_First_20_Tokens_Decoded"] = [record["decoded_text"] for record in literal_records]
    generation_df["Input_B_Generated_First_20_Token_IDs"] = [record["token_ids"] for record in metaphor_records]
    generation_df["Input_B_Generated_First_20_Token_Texts"] = [record["token_texts"] for record in metaphor_records]
    generation_df["Input_B_Generated_First_20_Tokens_Decoded"] = [record["decoded_text"] for record in metaphor_records]
    return generation_df


def run_pipeline(args: argparse.Namespace) -> None:
    language = normalize_language(args.language)
    model_id, model_name, target_layers, pca_layer, preset = resolve_model_preset(
        requested_model_id=args.model_id,
        requested_model_name=args.model_name,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        pca_layer=args.pca_layer,
    )

    output_dir = Path(args.output_root) / f"{sanitize_name(language)}_{sanitize_name(model_name)}_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("No Hugging Face token provided. Proceeding without explicit login.")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=args.device_map,
        torch_dtype=choose_torch_dtype(),
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    model.eval()
    input_device = get_input_device(model)

    print("Loading datasets...")
    experiment_df = pd.read_csv(args.experiment_csv)
    safe_lang_df = pd.read_csv(args.safe_lang_csv)
    safe_eng_df = pd.read_csv(args.safe_eng_csv)

    english_column = infer_column(experiment_df, args.english_column, ENGLISH_QUESTION_COLUMN_CANDIDATES, "unsafe English prompts")
    literal_column = infer_column(experiment_df, args.literal_column, LITERAL_COLUMN_CANDIDATES, "literal prompts")
    metaphor_column = infer_column(experiment_df, args.metaphor_column, METAPHOR_COLUMN_CANDIDATES, "metaphor prompts")
    safe_eng_column = infer_column(safe_eng_df, args.safe_eng_column, SAFE_PROMPT_COLUMN_CANDIDATES, "safe English prompts")
    safe_lang_column = infer_column(safe_lang_df, args.safe_lang_column, SAFE_LANG_COLUMN_CANDIDATES, "safe language prompts")
    harm_type_column = maybe_infer_column(experiment_df, args.harm_type_column, HARM_TYPE_COLUMN_CANDIDATES)

    safe_prompts_eng = safe_eng_df[safe_eng_column].dropna().astype(str).tolist()
    safe_prompts_lang = safe_lang_df[safe_lang_column].dropna().astype(str).tolist()
    unsafe_prompts_eng = experiment_df[english_column].dropna().astype(str).tolist()

    analysis_mask = experiment_df[literal_column].notna() & experiment_df[metaphor_column].notna()
    analysis_df = experiment_df.loc[analysis_mask].copy().reset_index().rename(columns={"index": "row_index"})
    if analysis_df.empty:
        raise ValueError("No rows contained both literal and metaphor prompts.")

    literal_prompts = analysis_df[literal_column].astype(str).tolist()
    metaphor_prompts = analysis_df[metaphor_column].astype(str).tolist()

    refusal_sequences = get_refusal_sequences(language)
    refusal_sequence_candidates = tokenize_candidate_sequences(tokenizer, refusal_sequences)

    print("Extracting layer-wise hidden states for safe vs unsafe English prompts...")
    safe_hiddens = get_last_token_hidden_states(
        safe_prompts_eng,
        target_layers,
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )
    unsafe_hiddens = get_last_token_hidden_states(
        unsafe_prompts_eng,
        target_layers,
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )

    print("Building refusal vectors with difference-in-means...")
    refusal_vectors: Dict[int, torch.Tensor] = {}
    for layer in target_layers:
        refusal_vectors[layer] = unsafe_hiddens[layer].mean(dim=0) - safe_hiddens[layer].mean(dim=0)

    print("Extracting hidden states for literal and metaphor prompts...")
    literal_hiddens = get_last_token_hidden_states(
        literal_prompts,
        target_layers,
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )
    metaphor_hiddens = get_last_token_hidden_states(
        metaphor_prompts,
        target_layers,
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )

    results_df = analysis_df[["row_index", literal_column, metaphor_column]].copy()
    if english_column in analysis_df.columns:
        results_df[english_column] = analysis_df[english_column]
    if harm_type_column and harm_type_column in analysis_df.columns:
        results_df[harm_type_column] = analysis_df[harm_type_column]

    print("Calculating layer-wise dot-product drift scores...")
    for layer in target_layers:
        ref_vec = refusal_vectors[layer]
        dot_a = literal_hiddens[layer] @ ref_vec
        dot_b = metaphor_hiddens[layer] @ ref_vec
        results_df[f"L{layer}_Dot_A"] = dot_a.numpy()
        results_df[f"L{layer}_Dot_B"] = dot_b.numpy()
        results_df[f"L{layer}_Dot_Drift"] = (dot_a - dot_b).numpy()

    print("Training layer-wise linear probes...")
    probe_summaries = []
    for layer in target_layers:
        probe_bundle, summary = train_linear_probe(
            safe_hiddens[layer],
            unsafe_hiddens[layer],
            layer=layer,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            val_fraction=args.probe_val_fraction,
            seed=args.seed,
        )
        probe_summaries.append(summary)

        logits_a, probs_a = apply_linear_probe(literal_hiddens[layer], probe_bundle)
        logits_b, probs_b = apply_linear_probe(metaphor_hiddens[layer], probe_bundle)

        results_df[f"L{layer}_Probe_Logit_A"] = logits_a.numpy()
        results_df[f"L{layer}_Probe_Logit_B"] = logits_b.numpy()
        results_df[f"L{layer}_Probe_Prob_A"] = probs_a.numpy()
        results_df[f"L{layer}_Probe_Prob_B"] = probs_b.numpy()
        results_df[f"L{layer}_Probe_Prob_Drift"] = (probs_a - probs_b).numpy()

    probe_df = pd.DataFrame(probe_summaries).sort_values("layer").reset_index(drop=True)
    print_probe_summary(probe_df)

    print("Evaluating sequence-level refusal CLL metrics...")
    seq_metrics_a = []
    seq_metrics_b = []
    for row_idx, row in analysis_df.iterrows():
        if row_idx % 10 == 0:
            print(f"  Processing sequence-level CLL for row {row_idx + 1}/{len(analysis_df)}")

        seq_metrics_a.append(
            get_sequence_level_refusal_metrics(
                str(row[literal_column]),
                refusal_sequence_candidates,
                tokenizer,
                model,
                input_device,
                batch_size=args.sequence_cll_batch_size,
            )
        )
        seq_metrics_b.append(
            get_sequence_level_refusal_metrics(
                str(row[metaphor_column]),
                refusal_sequence_candidates,
                tokenizer,
                model,
                input_device,
                batch_size=args.sequence_cll_batch_size,
            )
        )

    for key in (
        "best_refusal_sequence_avg",
        "best_refusal_sequence_total",
        "seq_cll_best_avg",
        "seq_cll_best_total",
        "seq_cll_any_refusal_logprob",
    ):
        results_df[f"A_{key}"] = [item[key] for item in seq_metrics_a]
        results_df[f"B_{key}"] = [item[key] for item in seq_metrics_b]

    print(f"Computing PCA and cross-language alignment at layer {pca_layer}...")
    pca_safe_eng = get_last_token_hidden_states(
        safe_prompts_eng,
        [pca_layer],
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )[pca_layer].numpy()
    pca_unsafe_eng = get_last_token_hidden_states(
        unsafe_prompts_eng,
        [pca_layer],
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )[pca_layer].numpy()
    pca_safe_lang = get_last_token_hidden_states(
        safe_prompts_lang,
        [pca_layer],
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )[pca_layer].numpy()
    pca_unsafe_lang_lit = get_last_token_hidden_states(
        literal_prompts,
        [pca_layer],
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )[pca_layer].numpy()
    pca_unsafe_lang_met = get_last_token_hidden_states(
        metaphor_prompts,
        [pca_layer],
        tokenizer,
        model,
        input_device,
        batch_size=args.hidden_batch_size,
    )[pca_layer].numpy()

    mu_safe_eng = np.mean(pca_safe_eng, axis=0)
    mu_unsafe_eng = np.mean(pca_unsafe_eng, axis=0)
    mu_safe_lang = np.mean(pca_safe_lang, axis=0)
    mu_unsafe_lang_lit = np.mean(pca_unsafe_lang_lit, axis=0)
    mu_unsafe_lang_met = np.mean(pca_unsafe_lang_met, axis=0)

    v_en = mu_unsafe_eng - mu_safe_eng
    v_lang_lit = mu_unsafe_lang_lit - mu_safe_lang
    v_lang_met = mu_unsafe_lang_met - mu_safe_lang

    dot_en_self = float(np.dot(v_en, v_en))
    dot_en_vs_lang_lit = float(np.dot(v_en, v_lang_lit))
    dot_en_vs_lang_met = float(np.dot(v_en, v_lang_met))

    alignment_summary_df = pd.DataFrame(
        [
            {"metric": "dot_product_english_self", "value": dot_en_self},
            {"metric": "dot_product_english_vs_literal", "value": dot_en_vs_lang_lit},
            {"metric": "dot_product_english_vs_metaphor", "value": dot_en_vs_lang_met},
            {
                "metric": "retained_refusal_magnitude_pct_literal",
                "value": float((dot_en_vs_lang_lit / dot_en_self) * 100) if dot_en_self else float("nan"),
            },
            {
                "metric": "retained_refusal_magnitude_pct_metaphor",
                "value": float((dot_en_vs_lang_met / dot_en_self) * 100) if dot_en_self else float("nan"),
            },
            {"metric": "cosine_similarity_english_vs_literal", "value": safe_cosine_similarity(v_en, v_lang_lit)},
            {"metric": "cosine_similarity_english_vs_metaphor", "value": safe_cosine_similarity(v_en, v_lang_met)},
            {"metric": "cosine_similarity_literal_vs_metaphor", "value": safe_cosine_similarity(v_lang_lit, v_lang_met)},
        ]
    )

    english_baseline_states = np.vstack([pca_safe_eng, pca_unsafe_eng])
    pca = PCA(n_components=2)
    pca.fit(english_baseline_states)

    pca_safe_eng_2d = pca.transform(pca_safe_eng)
    pca_unsafe_eng_2d = pca.transform(pca_unsafe_eng)
    pca_safe_lang_2d = pca.transform(pca_safe_lang)
    pca_unsafe_lang_lit_2d = pca.transform(pca_unsafe_lang_lit)
    pca_unsafe_lang_met_2d = pca.transform(pca_unsafe_lang_met)

    pca_3d = PCA(n_components=3)
    pca_3d.fit(english_baseline_states)

    pca_safe_eng_3d = pca_3d.transform(pca_safe_eng)
    pca_unsafe_eng_3d = pca_3d.transform(pca_unsafe_eng)
    pca_safe_lang_3d = pca_3d.transform(pca_safe_lang)
    pca_unsafe_lang_lit_3d = pca_3d.transform(pca_unsafe_lang_lit)
    pca_unsafe_lang_met_3d = pca_3d.transform(pca_unsafe_lang_met)

    print("Generating first 20 tokens for Input_A_Literal and Input_B_Metaphor...")
    generation_df = align_generation_outputs(
        analysis_df=analysis_df,
        literal_column=literal_column,
        metaphor_column=metaphor_column,
        tokenizer=tokenizer,
        model=model,
        input_device=input_device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.generation_batch_size,
    )

    join_drop_columns = [literal_column, metaphor_column]
    if english_column in results_df.columns:
        join_drop_columns.append(english_column)
    if harm_type_column and harm_type_column in results_df.columns:
        join_drop_columns.append(harm_type_column)

    full_results_df = experiment_df.copy().join(
        results_df.set_index("row_index").drop(columns=join_drop_columns, errors="ignore"),
        how="left",
    )

    main_results_path = artifact_path(output_dir, language, model_name, "layerwise_analysis", ".csv")
    probe_summary_path = artifact_path(output_dir, language, model_name, "linear_probe_summary", ".csv")
    generation_path = artifact_path(output_dir, language, model_name, "literal_metaphor_first20_generations", ".csv")
    alignment_summary_path = artifact_path(output_dir, language, model_name, "alignment_summary", ".csv")
    refusal_path = artifact_path(output_dir, language, model_name, "refusal_sequences", ".json")
    config_path = artifact_path(output_dir, language, model_name, "run_config", ".json")

    full_results_df.to_csv(main_results_path, index=False)
    probe_df.to_csv(probe_summary_path, index=False)
    generation_df.to_csv(generation_path, index=False)
    alignment_summary_df.to_csv(alignment_summary_path, index=False)
    save_json(refusal_path, {"language": language, "refusal_sequences": refusal_sequences})
    save_json(
        config_path,
        {
            "language": language,
            "model_id": model_id,
            "model_name": model_name,
            "target_layers": list(target_layers),
            "pca_layer": pca_layer,
            "preset": asdict(preset) if preset is not None else None,
            "paths": {
                "experiment_csv": args.experiment_csv,
                "safe_lang_csv": args.safe_lang_csv,
                "safe_eng_csv": args.safe_eng_csv,
                "output_dir": str(output_dir),
            },
            "columns": {
                "english_column": english_column,
                "literal_column": literal_column,
                "metaphor_column": metaphor_column,
                "safe_eng_column": safe_eng_column,
                "safe_lang_column": safe_lang_column,
                "harm_type_column": harm_type_column,
            },
        },
    )

    print("Saving charts...")
    try:
        set_plot_theme()
        save_internal_alignment_plot(
            results_df,
            target_layers,
            artifact_path(output_dir, language, model_name, "internal_alignment_with_refusal_direction_across_layers", ".png"),
            literal_column,
            metaphor_column,
        )
        save_dot_drift_bar_plot(
            results_df,
            target_layers,
            artifact_path(output_dir, language, model_name, "mean_dot_product_drift_by_layer", ".png"),
        )
        save_sequence_cll_scatter(
            results_df,
            artifact_path(output_dir, language, model_name, "sequence_level_refusal_cll_literal_vs_metaphor", ".png"),
        )
        save_probe_confidence_plot(
            results_df,
            target_layers,
            artifact_path(output_dir, language, model_name, "linear_probe_confidence_across_layers", ".png"),
        )
        save_probe_validation_plot(
            probe_df,
            target_layers,
            artifact_path(output_dir, language, model_name, "linear_probe_validation_performance", ".png"),
        )
        save_pca_projection_plot(
            pca_safe_eng_2d,
            pca_unsafe_eng_2d,
            pca_safe_lang_2d,
            pca_unsafe_lang_lit_2d,
            pca_unsafe_lang_met_2d,
            artifact_path(output_dir, language, model_name, "pca_projection_on_english_refusal_plane", ".png"),
            layer=pca_layer,
            language_label=language,
        )
        save_pca_projection_plot_3d(
            pca_safe_eng_3d,
            pca_unsafe_eng_3d,
            pca_safe_lang_3d,
            pca_unsafe_lang_lit_3d,
            pca_unsafe_lang_met_3d,
            artifact_path(output_dir, language, model_name, "pca_projection_on_english_refusal_plane_3d", ".html"),
            layer=pca_layer,
            language_label=language,
        )
        save_cosine_similarity_plot(
            alignment_summary_df,
            artifact_path(output_dir, language, model_name, "refusal_vector_cosine_similarity", ".png"),
            layer=pca_layer,
        )

        if harm_type_column and harm_type_column in results_df.columns:
            save_harm_category_plot(
                results_df,
                harm_type_column,
                artifact_path(output_dir, language, model_name, "refusal_strength_by_harm_category", ".png"),
            )
    except Exception as exc:
        print(f"Chart generation failed after CSV export: {exc}")

    print(f"Pipeline complete. Outputs saved to {output_dir}")
    print(f"Main layer-wise analysis CSV: {main_results_path}")
    print(f"Linear probe summary CSV: {probe_summary_path}")
    print(f"Literal/metaphor generation CSV: {generation_path}")
    print(f"Alignment summary CSV: {alignment_summary_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
