import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from eval_benchmark import run_benchmark

# Сколько набирает на бенчмарке модель со случайными весами
# Уровень шума для acc_calib
CHECKPOINT = "checkpoints/large_v16k_data2"
N_MODELS = 30


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(CHECKPOINT)
    config = AutoConfig.from_pretrained(CHECKPOINT)

    acc_calib = []

    for seed in range(N_MODELS):
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(config).to(device)

        acc_calib.append(run_benchmark(model, tokenizer, device)["summary"]["acc_calib"])

    print(sorted(acc_calib))

    Path("results/random_controls.json").write_text(json.dumps({"architecture": CHECKPOINT, "acc_calib": acc_calib}))


if __name__ == "__main__":
    main()
