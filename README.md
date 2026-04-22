# lmsys-chat-1000

Multi-turn conversation subsets of [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
and benchmarking scripts for testing OpenAI-compatible LLM inference services
(MaaS) with realistic conversation workloads.

## Contents

| File | Purpose |
|------|---------|
| `lmsys_10000.jsonl` | **Main dataset**: 1,925 complete conversations / 10,008 user turns |
| `lmsys_1000.jsonl` | Smoke-test subset: 191 complete conversations / 1,001 user turns |
| `lmsys_sample.py` | Sample a subset from the full LMSYS-Chat-1M |
| `maas_bench.py` | OpenAI-compatible MaaS API benchmark driver |
| `LICENSE` | Data follows upstream LMSYS license; code is MIT |

## Dataset details

| Metric | `lmsys_10000.jsonl` | `lmsys_1000.jsonl` |
|--------|---------------------|--------------------|
| Conversations | 1,925 (all complete, no truncation) | 191 (all complete) |
| Total user turns (= request count) | 10,008 | 1,001 |
| Avg turns / conversation | 5.20 | 5.24 |
| Turn distribution | 3: 699, 4: 392, 5: 245, 6: 162, 7: 129, 8+: 297 | 3: 61, 4: 40, 5: 25, 6: 22, 7: 16, 8+: 27 |
| File size | ~14 MB | ~1 MB |
| Language | English | English |

Each line is one complete conversation:

```json
{
  "conversation_id": "...",
  "model": "vicuna-13b",
  "turn": 5,
  "language": "English",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
```

## Usage

### Re-sample from the full dataset

```bash
pip install datasets
# First accept license at: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
huggingface-cli login
python3 lmsys_sample.py --target 10000 --output lmsys_10000.jsonl
```

### Benchmark an OpenAI-compatible MaaS API

```bash
# Main test (10K turns)
python3 maas_bench.py \
    --endpoint https://maas.example.com/v1 \
    --model qwen3-32b \
    --api-key sk-xxx \
    --dataset lmsys_10000.jsonl \
    --concurrency 50 \
    --output results.jsonl

# Smoke test (1K turns, for quick verification)
python3 maas_bench.py --dataset lmsys_1000.jsonl ... --limit 10
```

Script output includes TTFT / E2EL percentiles and a layer breakdown by
`turn_idx` (to observe prefix-cache gains on later turns).

See each script's top docstring for full options.

## Design notes

- **Session order is preserved.** Within a conversation, turns are sent
  sequentially (turn N's prompt contains turns 1..N-1 as prefix).
- **Cross-conversation concurrency** simulates multi-user load.
- **`--mode replay` (default)** reuses the dataset's original assistant
  replies as history for subsequent turns — prompts are byte-identical across
  runs, enabling direct metric comparison across different backends.
- **`--mode live`** feeds the MaaS's own generated replies into history —
  closer to real interaction but non-reproducible.

## License

- **Data** (`lmsys_*.jsonl`) inherits the
  [LMSYS-Chat-1M Dataset License](https://huggingface.co/datasets/lmsys/lmsys-chat-1m):
  non-commercial research use only, subject to OpenAI Terms of Use.
- **Code** (`*.py`) is MIT.

See [LICENSE](./LICENSE).

## Citation

If you use this data, please cite the LMSYS-Chat-1M paper:

```bibtex
@inproceedings{zheng2024lmsyschat1m,
  title={LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and others},
  booktitle={ICLR},
  year={2024}
}
```
