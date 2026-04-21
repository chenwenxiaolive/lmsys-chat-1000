#!/usr/bin/env python3
"""
maas_bench.py

用 LMSYS multi-turn conversation 数据集压测 OpenAI 兼容的 MaaS API。

核心：
  - 会话内 turn 串行（同一 conversation 的 A→B→C 顺序）
  - 会话间并发（多个 conversation 同时发，模拟多用户）
  - 第 N 轮 prompt = 前 N-1 轮完整历史 + 当前 user message（天然 prefix 累积）
  - 流式接收精确测 TTFT
  - 分层统计（按 turn_idx 和 prompt 长度）

模式：
  --mode replay（默认）：用 dataset 里原始 assistant 回复作下一轮历史
                         好处：两次运行的 prompt 完全一致，指标可比
  --mode live     ：用 MaaS 本次生成的 response 作下一轮历史
                         好处：更贴近真实交互，但可复现性弱

用法：
  python maas_bench.py \
      --endpoint https://maas.example.com/v1 \
      --model qwen3-32b \
      --api-key sk-xxx \
      --dataset lmsys_1000.jsonl \
      --concurrency 50 \
      --output results.jsonl
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import numpy as np


async def send_chat(session, endpoint, model, messages, api_key,
                    max_tokens, stream=True):
    """发送一次 /chat/completions，返回 ttft/total_time/completion"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }

    t0 = time.perf_counter()
    ttft = None
    completion = ""
    completion_tokens = 0

    try:
        async with session.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text[:300]}",
                        "ttft": None,
                        "total_time": time.perf_counter() - t0}

            if stream:
                async for line in resp.content:
                    if not line.strip() or not line.startswith(b"data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == b"[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            completion += delta
                            completion_tokens += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            else:
                data = await resp.json()
                ttft = time.perf_counter() - t0
                completion = data["choices"][0]["message"]["content"]
                completion_tokens = data.get("usage", {}).get(
                    "completion_tokens", 0)
    except Exception as e:
        return {"error": str(e), "ttft": None,
                "total_time": time.perf_counter() - t0}

    return {
        "ttft": ttft,
        "total_time": time.perf_counter() - t0,
        "completion_tokens": completion_tokens,
        "completion": completion,
    }


async def run_conversation(session, args, conv, sink):
    """按 turn 顺序串行发送单个 conversation 的所有 user 轮次"""
    conv_id = conv["conversation_id"]
    history = []
    turn_idx = 0
    msgs = conv["conversation"]
    i = 0

    while i < len(msgs):
        m = msgs[i]
        if m["role"] != "user":
            i += 1
            continue

        to_send = history + [m]
        r = await send_chat(session, args.endpoint, args.model,
                            to_send, args.api_key, args.max_tokens)
        r["conversation_id"] = conv_id
        r["turn_idx"] = turn_idx
        r["prompt_msg_count"] = len(to_send)
        r["prompt_chars"] = sum(len(x["content"]) for x in to_send)
        sink.append(r)

        history.append(m)
        if args.mode == "replay":
            # 下一轮历史用 dataset 原始 assistant
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                history.append(msgs[i + 1])
                i += 2
            else:
                i += 1
        else:  # live
            if r.get("completion") and not r.get("error"):
                history.append({"role": "assistant",
                                "content": r["completion"]})
            i += 1
        turn_idx += 1


async def run_all(args, convs):
    sem = asyncio.Semaphore(args.concurrency)
    sink = []
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def bounded(c):
            async with sem:
                await run_conversation(session, args, c, sink)

        t0 = time.perf_counter()
        await asyncio.gather(*[bounded(c) for c in convs])
        duration = time.perf_counter() - t0

    return sink, duration


def pct(arr, p):
    return sorted(arr)[int(len(arr) * p)] * 1000 if arr else 0


def report(results, duration):
    errors = sum(1 for r in results if r.get("error"))
    ok = [r for r in results if not r.get("error") and r["ttft"] is not None]

    print(f"\n=== MaaS Benchmark Results ===")
    print(f"总请求: {len(results)}, 成功: {len(ok)}, 错误: {errors}")
    print(f"持续时间: {duration:.1f}s, 吞吐: {len(ok)/duration:.2f} req/s")
    if not ok:
        return

    ttfts = [r["ttft"] for r in ok]
    e2els = [r["total_time"] for r in ok]
    print(f"\nTTFT (ms): avg {np.mean(ttfts)*1000:.0f}  "
          f"p50 {pct(ttfts,0.5):.0f}  p95 {pct(ttfts,0.95):.0f}  "
          f"p99 {pct(ttfts,0.99):.0f}")
    print(f"E2EL (ms): avg {np.mean(e2els)*1000:.0f}  "
          f"p50 {pct(e2els,0.5):.0f}  p95 {pct(e2els,0.95):.0f}  "
          f"p99 {pct(e2els,0.99):.0f}")

    # 按 turn_idx 分层：看 prefix cache 收益
    by_turn = defaultdict(list)
    for r in ok:
        by_turn[r["turn_idx"]].append(r)
    print(f"\n按 turn_idx 分层 TTFT（第 1 轮 vs 后续，差距越大说明 prefix cache 收益越高）：")
    for t in sorted(by_turn)[:8]:
        rs = by_turn[t]
        ts = [r["ttft"] for r in rs]
        print(f"  第 {t+1:>2d} 轮: n={len(rs):4d}  "
              f"TTFT avg {np.mean(ts)*1000:7.0f}ms  p99 {pct(ts,0.99):7.0f}ms")

    # 按 prompt 长度分层
    print(f"\n按 prompt 字符数分层 TTFT：")
    buckets = [(0, 500), (500, 1500), (1500, 3000),
               (3000, 6000), (6000, 15000), (15000, 10**9)]
    for lo, hi in buckets:
        rs = [r for r in ok if lo <= r["prompt_chars"] < hi]
        if rs:
            ts = [r["ttft"] for r in rs]
            label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
            print(f"  [{label:>12s} chars]: n={len(rs):4d}  "
                  f"TTFT avg {np.mean(ts)*1000:7.0f}ms")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True,
                   help="OpenAI 兼容 API 地址，如 https://api.example.com/v1")
    p.add_argument("--model", required=True, help="模型名")
    p.add_argument("--dataset", default="lmsys_1000.jsonl",
                   help="LMSYS 采样文件")
    p.add_argument("--api-key", default="", help="Bearer token，可选")
    p.add_argument("--concurrency", type=int, default=50,
                   help="并发 conversation 数（默认 50）")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="每轮 max output token（默认 128）")
    p.add_argument("--mode", choices=["replay", "live"], default="replay",
                   help="replay=用 dataset 的 assistant，live=用 MaaS 生成")
    p.add_argument("--output", default="maas_results.jsonl")
    p.add_argument("--limit", type=int, default=0,
                   help="只跑前 N 个 conversation（调试用，0 表示全部）")
    args = p.parse_args()

    with open(args.dataset) as f:
        convs = [json.loads(l) for l in f if l.strip()]
    if args.limit > 0:
        convs = convs[:args.limit]

    total_rounds = sum(c["turn"] for c in convs)
    print(f"加载 {len(convs)} conversation, {total_rounds} 轮请求")
    print(f"endpoint={args.endpoint}, model={args.model}")
    print(f"concurrency={args.concurrency}, mode={args.mode}, "
          f"max_tokens={args.max_tokens}")

    results, duration = asyncio.run(run_all(args, convs))
    report(results, duration)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            # 省略 completion 正文节省空间
            r_out = {k: v for k, v in r.items() if k != "completion"}
            f.write(json.dumps(r_out, ensure_ascii=False) + "\n")
    print(f"\n详细结果 -> {args.output}")


if __name__ == "__main__":
    main()
