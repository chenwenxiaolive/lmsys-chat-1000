#!/usr/bin/env python3
"""
maas_bench.py

用 LMSYS multi-turn conversation 数据集压测 OpenAI 兼容的 MaaS API。

调度模式（wave）：
  所有请求按 (turn_idx, conversation_index) 排序。
  第 1 轮 wave 同时发完所有 conversation 的第 1 个 user message；全部完成后
  才开始第 2 轮 wave，以此类推。每一轮 wave 内部用 semaphore 限制同时
  in-flight 的请求数为 --concurrency。

  这样：
  - 首轮 wave = 所有 "全新" prompt，不能命中外部 KV pool → 冷启动 TTFT
  - 第 2+ 轮 wave = 上一轮 KV 已在 pool，触发 prefix cache 命中 → 命中后 TTFT
  - 同一 conversation 的多轮 prompt 保持前缀累积关系（A→A+B→A+B+C）

采集：流式 TTFT、完整 E2EL、按 turn_idx 分层

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


def precompute_turns(conv):
    """
    把 conversation 拆成 [(user_msg, assistant_msg_or_None), ...]。
    对于只有 user 无 assistant 的尾巴，assistant_msg 为 None。
    """
    turns = []
    msgs = conv["conversation"]
    i = 0
    while i < len(msgs):
        if msgs[i]["role"] != "user":
            i += 1
            continue
        u = msgs[i]
        a = None
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            a = msgs[i + 1]
            i += 2
        else:
            i += 1
        turns.append((u, a))
    return turns


async def send_one_turn(session, args, conv, turn_idx, sink):
    """
    发送 conv 的第 turn_idx 轮。
    prompt = 前 turn_idx 轮的 (user + assistant) + 当前 user message。
    assistant 历史：replay 模式用 dataset 的；live 模式用 MaaS 生成的。
    """
    turns = conv["_turns"]
    history = []
    for i in range(turn_idx):
        u, a = turns[i]
        history.append(u)
        if args.mode == "replay":
            if a is not None:
                history.append(a)
        else:  # live：前一轮 MaaS 的 completion
            live_reply = conv.get("_live", {}).get(i)
            if live_reply:
                history.append({"role": "assistant", "content": live_reply})
            elif a is not None:
                # 万一 live 缺失（报错等），用 dataset 兜底
                history.append(a)

    current_user, _ = turns[turn_idx]
    messages = history + [current_user]

    r = await send_chat(session, args.endpoint, args.model,
                        messages, args.api_key, args.max_tokens)
    r["conversation_id"] = conv["conversation_id"]
    r["turn_idx"] = turn_idx
    r["prompt_msg_count"] = len(messages)
    r["prompt_chars"] = sum(len(m["content"]) for m in messages)
    sink.append(r)

    if args.mode == "live" and not r.get("error") and r.get("completion"):
        conv.setdefault("_live", {})[turn_idx] = r["completion"]


async def run_all(args, convs):
    """
    按 (turn_idx, conversation_index) 排序调度，并发由 --concurrency 控制。
    实现为 wave 式：依次处理第 1、2、3... 轮。
    每一轮内，所有 conversation 的请求按 conv_idx 顺序进入 semaphore 队列；
    同时最多 args.concurrency 个请求在 MaaS 上运行。
    某一轮全部完成后，才开始下一轮。
    """
    for c in convs:
        c["_turns"] = precompute_turns(c)

    sem = asyncio.Semaphore(args.concurrency)
    sink = []
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    max_turn = max(len(c["_turns"]) for c in convs)

    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()

        for turn_idx in range(max_turn):
            tasks = []
            for c in convs:
                if turn_idx >= len(c["_turns"]):
                    continue

                async def bounded(c=c, ti=turn_idx):
                    async with sem:
                        await send_one_turn(session, args, c, ti, sink)

                tasks.append(bounded())

            if not tasks:
                continue

            wave_start = time.perf_counter()
            await asyncio.gather(*tasks)
            wave_elapsed = time.perf_counter() - wave_start
            print(f"  Wave {turn_idx + 1:2d}: {len(tasks):5d} requests, "
                  f"{wave_elapsed:6.1f}s "
                  f"({len(tasks)/wave_elapsed:.1f} req/s)")

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
