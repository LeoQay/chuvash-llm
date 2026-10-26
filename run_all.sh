# Скрипт для воспроизведения, ни разу запускался, служит в качестве примера
# tiny/small/medium/large и т д придуманы из головы, просто как пресеты

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONUNBUFFERED=1

# Загружаем / варим данные
uv run python chuvash_llm/prepare_data.py
uv run python chuvash_llm/train_tokenizer.py
uv run python chuvash_llm/add_extra_data.py

# tiny (8.9M) -> small (34M) -> medium (69M)
uv run python chuvash_llm/train.py --preset tiny --vocab 16000 --tokens 50e6 --lr 2e-3
uv run python chuvash_llm/train.py --preset small --vocab 16000 --tokens 70e6 --lr 1e-3
uv run python chuvash_llm/train.py --preset medium --vocab 16000 --tokens 60e6 --lr 8e-4

# Словарь размера 8k
uv run python chuvash_llm/train.py --preset tiny --vocab 8000 --tokens 50e6 --lr 2e-3

# MoE vs dense на одинаковом числе активных параметров
uv run python chuvash_llm/train.py --preset tiny --vocab 16000 --tokens 20e6 --lr 2e-3 --name tiny_v16k_20m
uv run python chuvash_llm/train.py --preset tiny_moe --vocab 16000 --tokens 20e6 --lr 2e-3 --name tiny_moe_v16k_20m

# Отдельный venv для MoE запусков с экспериментальным torch 2.15:
# uv venv -p 3.14 .venv-t215
# uv pip install -p .venv-t215 torch==2.15.0 --index-url https://download.pytorch.org/whl/test/cpu
# uv pip install -p .venv-t215 transformers==5.20.2 tokenizers tensorboard numpy pandas pyarrow

.venv-t215/bin/python chuvash_llm/train.py \
  --preset tiny_moe \
  --experts_impl grouped \
  --vocab 16000 \
  --tokens 20e6 \
  --lr 2e-3 \
  --name tiny_moe_grouped_v16k_20m

# 6-7 частовые запуски по прогреву мака)
# Проверки моделей на примерно одних гпу часах обучения, на одном числе проученных токенов, MoE против dense

uv run python chuvash_llm/train.py \
  --preset large \
  --vocab 16000 \
  --data data/tok16000_v2 \
  --tokens 148e6 \
  --lr 6e-4 \
  --warmup 300 \
  --eval_every 500 \
  --name large_v16k_data2

.venv-t215/bin/python chuvash_llm/train.py \
  --preset large_moe \
  --experts_impl grouped \
  --vocab 16000 \
  --data data/tok16000_v2 \
  --tokens 116307360 \
  --lr 6e-4 \
  --warmup 300 \
  --eval_every 500 \
  --name large_moe_v16k_data2

.venv-t215/bin/python chuvash_llm/train.py \
  --preset large_moe \
  --experts_impl grouped \
  --vocab 16000 \
  --data data/tok16000_v2 \
  --tokens 148e6 \
  --lr 6e-4 \
  --warmup 300 \
  --eval_every 500 \
  --name large_moe_v16k_data2_148m

uv run python chuvash_llm/train.py \
  --preset xl \
  --vocab 16000 \
  --data data/tok16000_v2 \
  --tokens 148e6 \
  --lr 6e-4 \
  --warmup 300 \
  --eval_every 500 \
  --name xl_v16k_data2


# Бенчи

.venv-t215/bin/python chuvash_llm/eval_benchmark.py \
  checkpoints/{large_moe_v16k_data2_148m,xl_v16k_data2,large_v16k_data2,large_moe_v16k_data2,medium_v16k,small_v16k,tiny_v16k,tiny_v8k,tiny_v16k_20m,tiny_moe_v16k_20m,tiny_moe_grouped_v16k_20m} \
  2>&1 | tee logs/benchmark.log

uv run python chuvash_llm/random_controls.py \
  2>&1 | tee logs/random_controls.log
