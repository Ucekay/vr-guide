"""Benchmark generation speed of the vr-guide system prompt against a local vLLM server.

Usage:
    python scripts/bench_llm_prompt.py [--n 5] [--base-url http://localhost:8000/v1] [--tag mtp]

Sends the exact system prompt + worked example from docs/llm_prompt_design.md to a
running `vllm serve` OpenAI-compatible endpoint, streaming the response to measure
time-to-first-token (TTFT), input (prompt) tokens/sec, and output (decode) tokens/sec.
Results are appended to out/llm_bench.csv (one row per run) for later comparison.
"""

import argparse
import csv
import json
import os
import time
import urllib.request

ESSAY_MD = """当那列蒸汽火车从拉西奥塔车站的深处缓缓驶来，迎面冲向1895年巴黎咖啡馆里目瞪口呆的观众时，人类与影像的关系被永久改变了。1895年12月28日，卢米埃尔兄弟在巴黎大咖啡馆的"印度沙龙"举办了世界上第一场公开售票的电影放映会。《火车进站》与其他九部短片一同放映。卢米埃尔兄弟这50秒的奇迹不仅是技术的胜利——活动摄影机捕捉了真实运动——更是心理学的突破。固定机位、自然光、每秒16帧的朴素记录，却创造了纵深空间的幻觉，火车头仿佛要冲破银幕的边界。"""

RULES = """【规则,必须严格遵守】
1. 只能根据上面的"展板内容"回答。如果访客问的内容在展板内容里找不到依据,就诚实地说你不知道这个细节,不要用你自己的常识编造答案。
2. 如果访客问的是展板内容里没提到的"另一部电影",不要凭记忆回答,而是引导访客:"请您走到那幅海报前,我再为您讲解。"
3. 如果访客聊的是和电影完全无关的话题,礼貌地说明"我是这里的展览导览员",把话题带回当前这幅海报。
4. 你的回答会被朗读出来,所以:
   - 用3到5句口语化的短句,不要分段、不要用markdown格式(不要加粗、不要列表符号、不要标题)。
   - 不要重复访客的问题,直接说答案。
5. 不管访客用什么语言提问,你的回答都必须翻译成和访客提问完全相同的语言。
6. 最后,只输出一个JSON对象,不要有任何JSON以外的文字,格式如下:
{"answer": "<给访客听的回答原文,用访客提问的语言>", "grounded": <true或false>}
"""


def build_system_prompt(include_zh_title: bool = True, movie: str | None = None) -> str:
    """Assemble the full rdb_data (chapter, title(s), year, essay) per
    docs/llm_prompt_design.md section 2. `include_zh_title` toggles whether
    movie_title_zh is repeated in the heading (it's arguably redundant with
    essay_md, which usually names the film in Chinese already in prose).

    `movie` (Poster ID) loads the exhibit content from Postgres instead of the
    built-in sample; the default stays the sample so runs remain comparable with
    earlier rows in out/llm_bench.csv."""
    if movie:
        from vr_guide.content_db import load_exhibit_markdown
        rdb_markdown = load_exhibit_markdown(movie, include_zh_title=include_zh_title)
        return (
            '你是VRChat电影史博物馆的AI导览员。访客现在正站在一幅电影海报前,'
            '你的任务是根据下面提供的"展板内容",用口语回答访客的问题。\n\n'
            f'【展板内容】\n{rdb_markdown}\n\n{RULES}'
        )
    if include_zh_title:
        heading = "## 火车进站(L'Arrivée d'un train en gare de La Ciotat, 1895)"
    else:
        heading = "## L'Arrivée d'un train en gare de La Ciotat (1895)"
    rdb_markdown = f"# 章节1:电影诞生期 (1895-1910)\n\n{heading}\n\n{ESSAY_MD}"
    return (
        '你是VRChat电影史博物馆的AI导览员。访客现在正站在一幅电影海报前,'
        '你的任务是根据下面提供的"展板内容",用口语回答访客的问题。\n\n'
        f'【展板内容】\n{rdb_markdown}\n\n{RULES}'
    )


SYSTEM_PROMPT = build_system_prompt(include_zh_title=True)

QUESTIONS = [
    "この映画が初めて上映されたのはいつですか？",
    "导演后来还拍了什么作品？",
    "大都会はどんな映画ですか？",
]


def stream_chat(base_url: str, model: str, question: str, system_prompt: str = SYSTEM_PROMPT):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.7,
        "top_p": 0.80,
        "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first_token_t = None
    out_chars = 0
    usage = None
    chunks = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    out_chars += len(content)
                    chunks += 1

    t_end = time.perf_counter()
    ttft = (first_token_t - t0) if first_token_t else None
    total_time = t_end - t0
    decode_time = (t_end - first_token_t) if first_token_t else None

    out_tokens = usage.get("completion_tokens") if usage else None
    in_tokens = usage.get("prompt_tokens") if usage else None
    tok_per_s = (out_tokens / decode_time) if (out_tokens and decode_time) else None
    # TTFT covers prompt processing (prefill) + time to emit the first output token;
    # at batch size 1 this is dominated by prefill, so prompt_tokens/ttft approximates
    # input (prefill) tokens/sec.
    in_tok_per_s = (in_tokens / ttft) if (in_tokens and ttft) else None

    return {
        "question": question,
        "ttft_s": ttft,
        "total_time_s": total_time,
        "decode_time_s": decode_time,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "output_chars": out_chars,
        "input_tokens_per_s": in_tok_per_s,
        "tokens_per_s": tok_per_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    ap.add_argument("--n", type=int, default=1, help="repeat each question n times")
    ap.add_argument("--tag", default="", help="label for this run, stored in the CSV (e.g. 'mtp', 'no-mtp')")
    ap.add_argument("--out-csv", default="out/llm_bench.csv")
    ap.add_argument("--no-zh-title", action="store_true", help="drop movie_title_zh from the heading (test redundancy with essay_md)")
    ap.add_argument("--movie", default=None, help="Poster ID (movie_slug or id): load exhibit content from Postgres instead of the built-in sample")
    args = ap.parse_args()

    system_prompt = build_system_prompt(include_zh_title=not args.no_zh_title, movie=args.movie)

    results = []
    for q in QUESTIONS:
        for i in range(args.n):
            r = stream_chat(args.base_url, args.model, q, system_prompt)
            r["tag"] = args.tag
            r["run"] = i
            results.append(r)
            print(
                f"[{q[:20]:20s} #{i}] TTFT={r['ttft_s']:.3f}s  "
                f"in_tokens={r['input_tokens']}  in_tok/s={r['input_tokens_per_s']:.1f}  "
                f"decode={r['decode_time_s']:.3f}s  out_tokens={r['output_tokens']}  "
                f"out_tok/s={r['tokens_per_s']:.1f}" if r["tokens_per_s"] else
                f"[{q[:20]:20s} #{i}] (no usage stats) total={r['total_time_s']:.3f}s out_chars={r['output_chars']}"
            )

    valid = [r for r in results if r["tokens_per_s"]]
    if valid:
        avg_ttft = sum(r["ttft_s"] for r in valid) / len(valid)
        avg_tps = sum(r["tokens_per_s"] for r in valid) / len(valid)
        avg_in_tps = sum(r["input_tokens_per_s"] for r in valid) / len(valid)
        avg_tokens = sum(r["output_tokens"] for r in valid) / len(valid)
        avg_total = sum(r["total_time_s"] for r in valid) / len(valid)
        print(f"\n=== Average over {len(valid)} runs (tag={args.tag!r}) ===")
        print(f"TTFT: {avg_ttft:.3f}s")
        print(f"Input (prefill) throughput: {avg_in_tps:.1f} tok/s")
        print(f"Output (decode) throughput: {avg_tps:.1f} tok/s")
        print(f"Avg output tokens: {avg_tokens:.1f}")
        print(f"Avg total response time: {avg_total:.3f}s")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    file_exists = os.path.isfile(args.out_csv)
    fieldnames = [
        "tag", "run", "question", "ttft_s", "input_tokens", "input_tokens_per_s",
        "output_tokens", "tokens_per_s", "decode_time_s", "total_time_s",
    ]
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults appended to {args.out_csv}")


if __name__ == "__main__":
    main()
