# lmsys-chat-1000

A 1000-turn subset of [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
and benchmarking scripts for testing OpenAI-compatible LLM inference services
(MaaS) with realistic multi-turn conversation workloads.

## Contents

| File | Purpose |
|------|---------|
| `lmsys_1000.jsonl` | 191 complete multi-turn conversations (1001 user turns total) |
| `lmsys_sample.py` | Sample a subset from the full LMSYS-Chat-1M |
| `maas_bench.py` | OpenAI-compatible MaaS API benchmark driver |
| `LICENSE` | Data follows upstream LMSYS license; code is MIT |

## Dataset details

| Metric | Value |
|--------|-------|
| Conversations | 191 (all complete, no truncation) |
| Total user turns (= request count) | 1001 |
| Avg turns / conversation | 5.24 |
| Turn distribution | 3: 61, 4: 40, 5: 25, 6: 22, 7: 16, 8–18: 27 |
| Max prompt length (prefix-accumulated) | ~12K tokens |
| Language | English |

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
python3 lmsys_sample.py --target 1000 --output lmsys_1000.jsonl
```

### Benchmark an OpenAI-compatible MaaS API

```bash
python3 maas_bench.py \
    --endpoint https://maas.example.com/v1 \
    --model qwen3-32b \
    --api-key sk-xxx \
    --dataset lmsys_1000.jsonl \
    --concurrency 50 \
    --output results.jsonl
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

- **Data** (`lmsys_1000.jsonl`) inherits the
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
