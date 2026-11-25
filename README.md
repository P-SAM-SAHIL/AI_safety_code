# The Illusion of Cross-Lingual Safety in Low-Resource Languages

[![arXiv](https://img.shields.io/badge/arXiv-2608.11146-b31b1b.svg)](https://arxiv.org/abs/2608.11146)
[![Paper PDF](https://img.shields.io/badge/Paper-AfriDNA__Arxiv.pdf-blue.svg)](AfriDNA_Arxiv.pdf)

Official code repository for the paper **"The Illusion of Cross-Lingual Safety in Low-Resource Languages"** ([arXiv:2608.11146](https://arxiv.org/abs/2608.11146)).

> **Authors:** Abigail Oppong\*, P Sam Sahil\*, Tadesse Destaw Belay, Maryam Ibrahim Mukhtar, Esmael Ahmed Abdu, Tassallah Abdullahi, Jessica Oparebea, Saminu Mohammad Aliyu, Idris Abdulmumin, Abubakar Juma Chilala, Nicholaus Dismas Ladislaus, Alfred Malengo Kondoro, Lemofouet Valdini Douglace, Shamsuddeen Hassan Muhammad, and Seid Muhie Yimam  
> **Equal Contribution (\*):** `abigoppong@gmail.com`, `p.samsahil2003@gmail.com`

> [!WARNING]
> **Content Warning:** This research and associated prompts contain examples of potentially offensive, unsafe, or harmful content intended solely for AI safety research and evaluation.

---

## Overview

Safety alignment in large language models (LLMs) is predominantly developed in English under the assumption that safety safeguards generalize across languages. This repository provides the **Latent Geometric Framework** and downstream behavioral evaluation pipeline used to test cross-lingual safety transfer across four low-resource African languages—**Amharic**, **Twi**, **Hausa**, and **Swahili**—using **LoDNA**, a paired safety dataset derived from **Do-Not-Answer (DNA)** (Wang et al., 2024b) that contrasts **literal translations** against **culturally localized** harmful prompts (alongside community-curated English prompts from Rashid et al., 2026 adapted for Twi).

### Key Findings

- **Severe cross-lingual safety signal loss (RQ1):** Across nearly all language–model pairs at Layer 16, harmful target-language prompts retain **less than 10%** of the English refusal signal. The only partial exception is **Swahili in Llama-3.1-8B-Instruct** (`0.4626` literal / `0.4577` cultural cosine similarity; `27.31%` / `27.73%` Signed Retained Component).
- **Language-specific geometry rather than a universal harm manifold (RQ2):** Linear probes confirm that English safe vs. unsafe prompts are robustly separable in residual-stream hidden states, yet non-English prompts cluster in language-specific subspaces rather than mapping onto the English unsafe region.
- **Semantic alignment without safety routing (RQ3):** Literal and culturally localized prompts within each language remain tightly aligned (`0.95–0.996` cosine similarity) but diverge in layer-wise drift and fail to reliably trigger English refusal representations.
- **Architecture-specific failure modes & downstream breakdown (RQ4):** Across all evaluated prompts, the average probability of emitting any standard refusal sequence is **1.05%** for Literal inputs and **1.15%** for Cultural inputs—failing to initiate a refusal in over **98.8%** of cases—accompanied by repetition, hallucination, unintended language switching, unsafe compliance, and Amharic script/tokenization degradation.

---

## Repository Structure

```text
.
├── multilingual_refusal_pipeline.py   # End-to-end latent geometric, probe, CLL, PCA, and generation pipeline
├── refusal_sequences.py               # English and African-language refusal candidate sequences & language aliases
├── AfriDNA_Arxiv.pdf                  # Full paper manuscript
└── README.md                          # Project documentation
```

---

## Supported Languages & Models

### Languages

Language normalization and language-specific refusal candidates are configured in [`refusal_sequences.py`](refusal_sequences.py):

| Language | Accepted `--language` Values | Family | Country | Script | Safe Baseline Source (Appendix A.1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Amharic** | `amh`, `amharic`, `am` | Afro-Asiatic (Semitic) | Ethiopia | Ge'ez | [`Henok/aya_amharic_dataset`](https://huggingface.co/datasets/Henok/aya_amharic_dataset) |
| **Twi** | `twi`, `tw` | Niger-Congo (Kwa) | Ghana | Latin | [`Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT`](https://huggingface.co/datasets/Ghana-NLP/ENGLISH_TWI_PARALLEL_TEXT) |
| **Hausa** | `hausa`, `hau`, `ha` | Afro-Asiatic (Chadic) | Nigeria | Latin | [`CohereForAI/aya_dataset`](https://huggingface.co/datasets/CohereForAI/aya_dataset) |
| **Swahili** | `swa`, `swahili`, `sw` | Niger-Congo (Bantu) | Tanzania | Latin | [`CohereForAI/aya_dataset`](https://huggingface.co/datasets/CohereForAI/aya_dataset) |
| **English (Control)** | — | — | — | Latin | [`SalKhan12/prompt-safety-dataset`](https://huggingface.co/datasets/SalKhan12/prompt-safety-dataset) |

### Model Presets

The following open-weight 7B–8B presets are built into [`multilingual_refusal_pipeline.py`](multilingual_refusal_pipeline.py):

| Model ID | Aliases | Output Name | Total Layers | Target Layers | PCA Layer | Prompt Formatting |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistral`, `mistral7b` | `mistral` | 32 | 14–24 | 16 | Chat template |
| `Qwen/Qwen2.5-7B-Instruct` | `qwen`, `qwen2.5` | `qwen2_5` | 28 | 12–20 | 16 | Chat template |
| `meta-llama/Llama-3.1-8B-Instruct` | `llama`, `llama31` | `llama` | 32 | 14–24 | 16 | Chat template |
| `McGill-NLP/AfriqueQwen-8B` | `afriqueqwen`, `afrique` | `afriqueqwen` | 36 | 16–28 | 20 | Base (`Question: ...\nAnswer: `), `<think>` tokens banned |

You can also run any other Hugging Face causal LM by supplying `--layer-start`, `--layer-end`, and `--pca-layer` explicitly (and optionally `--model-name` and `--trust-remote-code`).

---

## Installation

Python 3.9+ and a CUDA-capable GPU (recommended for 7B–8B `float16` inference) are required. Install dependencies with:

```bash
pip install torch transformers accelerate huggingface_hub numpy pandas scikit-learn matplotlib seaborn plotly
```

> [!NOTE]
> `plotly` is optional and only required to export the interactive 3D PCA HTML visualization (`*_pca_projection_on_english_refusal_plane_3d.html`). If not installed, the pipeline skips the 3D plot and completes all other outputs.

For gated Hugging Face models (such as `meta-llama/Llama-3.1-8B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3`), authenticate via environment variable or `--hf-token`:

```bash
export HF_TOKEN="your_huggingface_token"
```

---

## Input Data Format

[`multilingual_refusal_pipeline.py`](multilingual_refusal_pipeline.py) expects three CSV files (`--experiment-csv`, `--safe-lang-csv`, and `--safe-eng-csv`). Column names are automatically detected from common candidates or can be overridden via CLI flags:

1. **Experiment CSV (`--experiment-csv`)**: Paired harmful prompts per language (rows missing any of the three prompt columns are filtered out):
   - **English unsafe prompt:** `English Question`, `English_Question`, `english_question`, `English prompt`, `english_prompt` (override: `--english-column`)
   - **Literal translation (`Input A`):** `Input_A_Literal`, `Input A Literal`, `literal`, `Literal` (override: `--literal-column`)
   - **Cultural context (`Input B`):** `Input_B_Cultural_Context`, `Input B Cultural Context`, `cultural_context`, `Cultural_Context` (override: `--cultural-context-column`)
   - **Harm category *(optional, for grouped plots)*:** `types_of_harm`, `Types_of_Harm`, `type_of_harm`, `harm_type`, `Harm_Type` (override: `--harm-type-column`)

2. **Safe Target-Language CSV (`--safe-lang-csv`)**: Safe control prompts in the target language (truncated to match the number of paired unsafe prompts):
   - Auto-detected from `inputs`, `input`, `safe_prompt`, `Safe_Prompt`, `Safe Prompt`, `prompt`, `Prompt`, `text`, `Text`, `Safe_Control_Prompt`, `label`, `labels` (override: `--safe-lang-column`)

3. **Safe English CSV (`--safe-eng-csv`)**: Safe English control prompts (truncated to match the number of paired unsafe prompts):
   - Auto-detected from `Safe_Control_Prompt`, `safe_control_prompt`, `Safe Prompt`, `safe_prompt`, `prompt`, `Prompt`, `inputs`, `text` (override: `--safe-eng-column`)

---

## Usage

### Example Commands

**Amharic on Mistral-7B-Instruct-v0.3:**

```bash
python multilingual_refusal_pipeline.py \
  --language amh \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --experiment-csv path/to/amh_ready_for_experiment.csv \
  --safe-lang-csv path/to/amh_safe.csv \
  --safe-eng-csv path/to/Safe_prompts_eng.csv \
  --output-root ./results
```

**Twi on Llama-3.1-8B-Instruct:**

```bash
python multilingual_refusal_pipeline.py \
  --language twi \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --experiment-csv path/to/twi_ready_for_experiment.csv \
  --safe-lang-csv path/to/twi_safe.csv \
  --safe-eng-csv path/to/Safe_prompts_eng.csv \
  --output-root ./results
```

**Hausa on Qwen2.5-7B-Instruct:**

```bash
python multilingual_refusal_pipeline.py \
  --language hausa \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --experiment-csv path/to/hausa_ready_for_experiment.csv \
  --safe-lang-csv path/to/hausa_safe.csv \
  --safe-eng-csv path/to/Safe_prompts_eng.csv \
  --output-root ./results
```

**Swahili on AfriqueQwen-8B (using preset alias):**

```bash
python multilingual_refusal_pipeline.py \
  --language swa \
  --model-id afriqueqwen \
  --experiment-csv path/to/swa_ready_for_experiment.csv \
  --safe-lang-csv path/to/swa_safe.csv \
  --safe-eng-csv path/to/Safe_prompts_eng.csv \
  --output-root ./results
```

**Custom Hugging Face model with explicit layer settings:**

```bash
python multilingual_refusal_pipeline.py \
  --language swa \
  --model-id org/custom-model \
  --model-name custom_model \
  --layer-start 14 \
  --layer-end 24 \
  --pca-layer 16 \
  --experiment-csv path/to/swa_ready_for_experiment.csv \
  --safe-lang-csv path/to/swa_safe.csv \
  --safe-eng-csv path/to/Safe_prompts_eng.csv \
  --output-root ./results
```

<details>
<summary>All CLI Options</summary>

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--language` | *(required)* | Target language code or name (`amh`, `twi`, `hausa`/`hau`, `swa`, etc.) |
| `--model-id` | *(required)* | Hugging Face model ID or preset alias (`mistral`, `qwen`, `afriqueqwen`, `llama`) |
| `--model-name` | preset / sanitized ID | Short model name used in output directory and artifact filenames |
| `--experiment-csv` | *(required)* | Path to `{lang}_ready_for_experiment.csv` |
| `--safe-lang-csv` | *(required)* | Path to `{lang}_safe.csv` |
| `--safe-eng-csv` | *(required)* | Path to `Safe_prompts_eng.csv` |
| `--output-root` | `.` | Root directory where `<lang>_<model_name>_output/` will be created |
| `--english-column`, `--literal-column`, `--cultural-context-column` | auto-detected | Override experiment CSV prompt column names |
| `--safe-eng-column`, `--safe-lang-column`, `--harm-type-column` | auto-detected | Override safe prompt or harm-category column names |
| `--layer-start`, `--layer-end`, `--pca-layer` | from preset | Custom hidden-layer range and PCA layer (all three required for non-preset models) |
| `--hidden-batch-size` | `32` | Batch size for hidden-state extraction |
| `--generation-batch-size` | `32` | Batch size for greedy token generation |
| `--sequence-cll-batch-size` | `32` | Batch size for candidate refusal sequence log-likelihood evaluation |
| `--max-new-tokens` | `20` | Number of tokens generated per prompt |
| `--probe-epochs` | `200` | Training epochs for layer-wise English linear probes |
| `--probe-lr` | `1e-2` | AdamW learning rate for linear probes |
| `--probe-weight-decay` | `1e-4` | AdamW weight decay for linear probes |
| `--probe-val-fraction` | `0.2` | Validation split fraction for linear probes |
| `--seed` | `42` | Random seed (`torch` and `numpy`) |
| `--hf-token` | `$HF_TOKEN` / `$HUGGINGFACE_TOKEN` | Optional Hugging Face access token |
| `--device-map` | `auto` | Device map passed to `AutoModelForCausalLM.from_pretrained` |
| `--trust-remote-code` | `False` | Enable `trust_remote_code` when loading custom models/tokenizers |

</details>

---

## Pipeline Outputs

Results are written to `<output_root>/<lang>_<model_name>_output/` with filenames prefixed by `<prefix> = <lang>_<model_name>`:

### Tabular & Config Artifacts
- `<prefix>_layerwise_analysis.csv` — Per-prompt layer-wise dot products (`L{k}_Dot_A`, `L{k}_Dot_B`), drift (`L{k}_Dot_Drift`), zero-shot probe logits/probabilities/drift, and sequence-level refusal CLL metrics (`seq_cll_best_avg`, `seq_cll_best_total`, `seq_cll_any_refusal_logprob`).
- `<prefix>_linear_probe_summary.csv` — Layer-wise English linear probe training/validation accuracy and AUROC.
- `<prefix>_alignment_summary.csv` — PCA-layer dot products, **Signed Retained Component (%)**, and pairwise cosine similarities (`English vs Literal`, `English vs Cultural Context`, `Literal vs Cultural Context`).
- `<prefix>_literal_cultural_context_first20_generations.csv` — Greedy first-`N` (default 20) generated token IDs, token texts, and decoded responses for English, Literal (`Input A`), and Cultural Context (`Input B`) prompts.
- `<prefix>_refusal_sequences.json` — Candidate English and target-language refusal sequences used for CLL/SLL scoring.
- `<prefix>_run_config.json` — Full run configuration, resolved model preset, layer range, input paths, and inferred column names.

### Visualizations
- `<prefix>_internal_alignment_with_refusal_direction_across_layers.png`
- `<prefix>_mean_dot_product_drift_by_layer.png`
- `<prefix>_sequence_level_refusal_cll_literal_vs_cultural_context.png`
- `<prefix>_linear_probe_confidence_across_layers.png`
- `<prefix>_linear_probe_validation_performance.png`
- `<prefix>_pca_projection_on_english_refusal_plane.png`
- `<prefix>_pca_projection_on_english_refusal_plane_3d.html` *(if `plotly` is installed)*
- `<prefix>_refusal_vector_cosine_similarity.png`
- `<prefix>_refusal_strength_by_harm_category.png` *(if a harm-type column is present)*

---

## Summary of Results (from the Paper)

### Cross-Lingual Safety Alignment at Layer 16 (Table 2)

| Model | Lang | Cos (Eng vs Lit) | Cos (Eng vs Ctx) | Cos (Lit vs Ctx) | Dot Eng (Self) | Dot Lit | Dot Ctx | Retained Comp: Lit | Retained Comp: Ctx | Mean Drift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mistral** | AMH | 0.0732 | 0.0657 | 0.9763 | 5.9656 | 0.1263 | 0.1156 | 2.1167% | 1.9374% | 0.0107 |
| **Mistral** | HAU | -0.0162 | -0.0743 | 0.9700 | 6.5436 | -0.0391 | -0.1892 | -0.5970% | -2.8911% | 0.1501 |
| **Mistral** | SWA | -0.1482 | -0.1489 | 0.9709 | 6.0441 | -0.4487 | -0.4674 | -7.4245% | -7.7326% | 0.0187 |
| **Mistral** | TWI | 0.0762 | 0.0505 | 0.9895 | 4.7016 | 0.1730 | 0.1117 | 3.6790% | 2.3766% | 0.0613 |
| **Llama** | AMH | -0.1456 | -0.1423 | 0.9925 | 15.5597 | -1.1578 | -1.1805 | -7.4410% | -7.5866% | 0.0227 |
| **Llama** | HAU | -0.0276 | -0.0649 | 0.9841 | 15.2320 | -0.2400 | -0.5651 | -1.5758% | -3.7099% | 0.3251 |
| **Llama** | SWA | **0.4626** | **0.4577** | 0.9946 | 17.3232 | 4.7303 | 4.8046 | **27.3063%** | **27.7349%** | -0.0743 |
| **Llama** | TWI | 0.1212 | 0.1019 | 0.9914 | 14.4152 | 0.8430 | 0.7122 | 5.8478% | 4.9406% | 0.1308 |
| **Qwen 2.5** | AMH | -0.0436 | 0.0168 | 0.9740 | 319.1111 | -3.8811 | 1.5546 | -1.2162% | 0.4872% | -5.4357 |
| **Qwen 2.5** | HAU | 0.0323 | -0.0716 | 0.9518 | 311.9406 | 3.0291 | -6.8713 | 0.9710% | -2.2028% | 9.9004 |
| **Qwen 2.5** | SWA | -0.0215 | -0.0468 | 0.9843 | 325.3607 | -3.4486 | -7.3608 | -1.0599% | -2.2623% | 3.9122 |
| **Qwen 2.5** | TWI | 0.1430 | 0.1105 | 0.9957 | 233.8068 | 20.0437 | 15.3946 | 8.5728% | 6.5843% | 4.6491 |

### Downstream Normalized Refusal SLL & Refusal Probability (Table 4)

| Group | SLL A (Literal) | SLL B (Cultural) | Drift (B − A) | Ref. Prob A / B (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Amh** | -1.280 | -1.294 | -0.013 | 0.18 / 0.16 |
| **Hau** | -1.563 | -1.550 | +0.012 | 3.57 / 3.95 |
| **Swa** | -1.742 | -1.757 | -0.015 | 0.22 / 0.24 |
| **Twi** | -2.512 | -2.506 | +0.006 | 0.23 / 0.23 |
| **Llama** | -1.280 | -1.269 | +0.011 | 1.83 / 2.23 |
| **Mistral** | -1.432 | -1.476 | -0.044 | 1.00 / 0.87 |
| **Qwen2.5** | -2.610 | -2.585 | +0.025 | 0.32 / 0.33 |
| **Overall** | **-1.774** | **-1.777** | **-0.002** | **1.05 / 1.15** |

---

## Ethical Use

Consistent with Section 8 of the paper, this work extends publicly available safety benchmarks with native-speaker translations and cultural localizations using no personally identifiable information, and is released **for research purposes only**.

---

## Citation

If you find this repository or paper useful in your research, please cite:

```bibtex
@misc{oppong2026illusioncrosslingualsafetylowresource,
      title={The Illusion of Cross-Lingual Safety in Low-Resource Languages}, 
      author={Abigail Oppong and P Sam Sahil and Tadesse Destaw Belay and Maryam Ibrahim Mukhtar and Esmael Ahmed Abdu and Tassallah Abdullahi and Jessica Oparebea and Saminu Mohammad Aliyu and Idris Abdulmumin and Abubakar Juma Chilala and Nicholaus Dismas Ladislaus and Alfred Malengo Kondoro and Lemofouet Valdini Douglace and Shamsuddeen Hassan Muhammad and Seid Muhie Yimam},
      year={2026},
      eprint={2608.11146},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.11146}, 
}
```
