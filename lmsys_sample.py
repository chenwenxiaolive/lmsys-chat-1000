#!/usr/bin/env python3
"""
lmsys_sample.py

从 LMSYS-Chat-1M 采样子集，保留完整 conversation：
  - 每条输出 = 一个完整 conversation（所有 user/assistant 轮次都保留）
  - 总请求数（= 所有 conversation 的 user 轮数之和）约等于 --target
  - 按原始顺序取，直到累计请求数达到目标

用法：
  # 方式 1：直接从 HuggingFace 加载（需先 huggingface-cli login 并接受 license）
  python lmsys_sample.py --target 1000 --output lmsys_1000.jsonl

  # 方式 2：从本地 parquet 读取
  python lmsys_sample.py --local-file /path/to/train-00000-of-*.parquet \\
      --target 1000 --output lmsys_1000.jsonl

输出格式（每行一个 conversation）：
  {
    "conversation_id": "...",
    "model": "...",
    "turn": 3,                // user 轮数（即该 conversation 的 request 数）
    "language": "English",
    "conversation": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]
  }
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_from_hf(source, split):
    try:
        from datasets import load_dataset
    except ImportError:
        print("缺少 datasets 库，请先 `pip install datasets`", file=sys.stderr)
        sys.exit(1)
    try:
        return load_dataset(source, split=split)
    except Exception as e:
        print(f"HF 加载失败: {e}", file=sys.stderr)
        print("可能原因：未 huggingface-cli login 或未接受 license。", file=sys.stderr)
        print("接受 license: https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
              file=sys.stderr)
        sys.exit(1)


def load_from_local(path):
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            print("缺少 pandas，请先 `pip install pandas pyarrow`", file=sys.stderr)
            sys.exit(1)
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    if path.suffix in (".jsonl", ".json"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"不支持的后缀: {path.suffix}（需 .parquet / .jsonl）")


def count_user_turns(conversation):
    return sum(1 for m in conversation if m.get("role") == "user")


def filter_record(rec, min_turns, max_turns, language):
    conv = rec.get("conversation")
    if not conv:
        return False
    if language and rec.get("language") != language:
        return False
    u = count_user_turns(conv)
    return min_turns <= u <= max_turns


def sample(records, target, min_turns, max_turns, language):
    selected = []
    total = 0
    for rec in records:
        if not filter_record(rec, min_turns, max_turns, language):
            continue
        u = count_user_turns(rec["conversation"])
        selected.append(rec)
        total += u
        if total >= target:
            break
    return selected, total


def normalize_conversation(conv):
    """
    把 conversation 规范化成 [{"role": "...", "content": "..."}] 列表。
    LMSYS 原始字段本身已经是这个格式，这里做容错。
    """
    result = []
    for m in conv:
        if isinstance(m, dict) and "role" in m and "content" in m:
            result.append({"role": m["role"], "content": m["content"]})
    return result


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=1000,
                   help="目标总请求数（conversation 的 user 轮次之和，默认 1000）")
    p.add_argument("--output", required=True, help="输出 jsonl 路径")
    p.add_argument("--source", default="lmsys/lmsys-chat-1m",
                   help="HF 数据集名（默认 lmsys/lmsys-chat-1m）")
    p.add_argument("--split", default="train", help="HF split（默认 train）")
    p.add_argument("--local-file", help="本地 parquet 或 jsonl 文件路径")
    p.add_argument("--language", default="English",
                   help="筛选语言（默认 English，空字符串 '' 表示不过滤）")
    p.add_argument("--min-turns", type=int, default=3,
                   help="最少 user 轮数（默认 3，聚焦多轮对话）")
    p.add_argument("--max-turns", type=int, default=20,
                   help="最多 user 轮数（默认 20，避免过长会话）")
    args = p.parse_args()

    if args.local_file:
        print(f"从本地加载: {args.local_file}")
        records = load_from_local(args.local_file)
    else:
        print(f"从 HuggingFace 加载: {args.source} (split={args.split})")
        records = load_from_hf(args.source, args.split)
    print(f"  总 records: {len(records)}")

    language = args.language or None
    selected, total = sample(records, args.target,
                              args.min_turns, args.max_turns, language)

    print(f"\n筛选条件: language={language!r}, "
          f"user_turns ∈ [{args.min_turns}, {args.max_turns}]")
    print(f"选中 conversation: {len(selected)}")
    print(f"总请求数 (user turns): {total}")
    if selected:
        print(f"平均每 conversation: {total/len(selected):.2f} 轮")
        dist = Counter(count_user_turns(r["conversation"]) for r in selected)
        parts = " ".join(f"{k}轮={dist[k]}" for k in sorted(dist))
        print(f"轮次分布: {parts}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in selected:
            conv = normalize_conversation(rec["conversation"])
            f.write(json.dumps({
                "conversation_id": rec.get("conversation_id", ""),
                "model": rec.get("model", ""),
                "turn": count_user_turns(conv),
                "language": rec.get("language", ""),
                "conversation": conv,
            }, ensure_ascii=False) + "\n")
    print(f"\n输出 -> {out}")


if __name__ == "__main__":
    main()
