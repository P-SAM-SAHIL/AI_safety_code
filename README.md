# AI_safety_code

! python run_experiment.py \
  --language amh \
  --model-id google/gemma-3-12b-it \
  --experiment-csv amh_ready_for_experiment.csv \
  --safe-lang-csv amh_safe.csv \
  --safe-eng-csv Safe_prompts_eng.csv


  python run_experiment.py \
  --language twi \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --experiment-csv twi_ready_for_experiment.csv \
  --safe-lang-csv twi_safe.csv \
  --safe-eng-csv Safe_prompts_eng.csv

  python run_experiment.py \
  --language hausa \
  --model-id Qwen/Qwen3-8B \
  --experiment-csv hausa_ready_for_experiment.csv \
  --safe-lang-csv hausa_safe.csv \
  --safe-eng-csv Safe_prompts_eng.csv \
  --output-root /path/to/my/results_folder
