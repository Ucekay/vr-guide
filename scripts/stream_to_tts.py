"""Stream LLM output into a TTS engine, chunked at clause boundaries.

Usage:
    python scripts/stream_to_tts.py "この映画が初めて上映されたのはいつですか？"

Design (see docs/llm_prompt_design.md, 2026-10-01 "structured output vs streaming" decision):
  - The synchronous, user-facing path asks the model for PLAIN TEXT ONLY (no JSON).
    Structured metadata (e.g. `grounded`) is pushed to a separate, non-blocking call
    (see `check_grounded_async`) so it never costs the visitor latency. This mirrors
    the production pattern documented by AssemblyAI's voice-pipeline guide: classify
    separately, stream the conversational reply.
  - Tokens are buffered and flushed to TTS at clause boundaries (、，。！？), not
    sentence boundaries, matching CosyVoice's own internal streaming chunk size
    (~15 speech tokens per the CosyVoice2 paper) rather than waiting for a full
    sentence. A minimum chunk length guards against feeding TTS tiny fragments
    (per the arXiv 2603.06444 finding that near-zero context spikes WER).
"""

import argparse
import json
import sys
import time
import urllib.request

# movie_title_zh is deliberately NOT included here (2026-10-01 decision): supporting
# "what's the Chinese title?" would also demand English/Korean/etc. title fields,
# which doesn't scale. title_orig + the existing "translate the answer into the
# visitor's question language" rule already covers "what's this called in <language>?"
# for any language, with no per-language schema field needed.
RDB_MARKDOWN = """# 章节1:电影诞生期 (1895-1910)

## L'Arrivée d'un train en gare de La Ciotat (1895)

当那列蒸汽火车从拉西奥塔车站的深处缓缓驶来，迎面冲向1895年巴黎咖啡馆里目瞪口呆的观众时，人类与影像的关系被永久改变了。1895年12月28日，卢米埃尔兄弟在巴黎大咖啡馆的"印度沙龙"举办了世界上第一场公开售票的电影放映会。《火车进站》与其他九部短片一同放映。卢米埃尔兄弟这50秒的奇迹不仅是技术的胜利——活动摄影机捕捉了真实运动——更是心理学的突破。固定机位、自然光、每秒16帧的朴素记录，却创造了纵深空间的幻觉，火车头仿佛要冲破银幕的边界。"""

# 2026-10-08: reordered so the parts that NEVER change (persona + rules) come
# first and the part that changes per-movie (exhibit content) comes last, right
# before the per-turn conversation history/question. vLLM's prefix cache only
# reuses a PREFIX match: with exhibit content first (the old order), two
# visitors looking at different posters diverge immediately, so even the
# byte-identical rules text further down never hits the cache. With rules
# first, every request to this server — regardless of which movie it's about —
# shares one cached prefix, which matters precisely because this is a
# server-centralized, multi-visitor deployment (ADR 0001).
#
# Plain-text-only variant: the old rule 6 (JSON output) was removed. This is
# the synchronous, user-facing prompt. `grounded` moved entirely to
# check_grounded_async().
def system_prompt_zh(rdb_markdown: str) -> str:
    return f"""你是VRChat电影史博物馆的AI导览员。访客现在正站在一幅电影海报前,你的任务是根据下面提供的"展板内容",用口语回答访客的问题。

【规则,必须严格遵守】
1. 只能根据下面的"展板内容"回答。如果访客问的内容在展板内容里找不到依据,就诚实地说你不知道这个细节,不要用你自己的常识编造答案。
2. 如果访客问的是展板内容里没提到的"另一部电影",不要凭记忆回答,而是引导访客:"请您走到那幅海报前,我再为您讲解。"
3. 如果访客聊的是和电影完全无关的话题,礼貌地说明"我是这里的展览导览员",把话题带回当前这幅海报。
4. 你的回答会被朗读出来,所以:
   - 用3到5句口语化的短句,不要分段、不要用markdown格式(不要加粗、不要列表符号、不要标题)。
   - 不要重复访客的问题,直接说答案。
5. 不管访客用什么语言提问,你的回答都必须翻译成和访客提问完全相同的语言。
6. 直接输出给访客听的回答原文,不要输出任何JSON、标签或其他包装。

【展板内容】
{rdb_markdown}
"""

# A/B variant from the earlier prompt-language test: instructions (rules) in
# English, exhibit content left in Chinese as-is. Won that A/B test (see the
# note on SYSTEM_PROMPT below), so this is now the default.
def system_prompt_en(rdb_markdown: str) -> str:
    return f"""You are the AI docent of a VRChat film-history museum. The visitor is standing in front of a movie poster right now. Answer the visitor's question using only the "exhibit content" provided below, speaking in a natural spoken style.

[RULES — follow strictly]
1. Answer only based on the "exhibit content" below. If the visitor's question isn't covered there, honestly say you don't know that detail — never fill the gap with general knowledge.
2. If the visitor asks about a different movie not covered in the exhibit content, don't answer from memory — instead guide them: "Please walk over to that poster, and I'll explain it there."
3. If the visitor brings up a topic unrelated to film entirely, politely explain "I'm the guide for this exhibit" and steer the conversation back to the current poster.
4. Your answer will be read aloud, so:
   - Use 3 to 5 short, conversational sentences. No paragraphs, no markdown formatting (no bold, no bullet points, no headings).
   - Don't repeat the visitor's question — answer directly.
5. No matter what language the visitor asks in, your answer must be translated into that exact same language.
6. Output only the spoken answer text itself — no JSON, no tags, no other wrapper.

[EXHIBIT CONTENT]
{rdb_markdown}
"""

SYSTEM_PROMPT_ZH = system_prompt_zh(RDB_MARKDOWN)
SYSTEM_PROMPT_EN = system_prompt_en(RDB_MARKDOWN)

# 2026-10-08: switched default to English rules after an A/B test showed the
# Chinese-rules prompt misclassifying "what did the director do later?" (not
# covered by the exhibit content) as "a different movie" and wrongly invoking
# the redirect rule, while the English-rules prompt correctly said it didn't
# know and only then suggested the redirect. See out/ or this session's log
# for the raw transcripts; not yet captured in a committed file.
SYSTEM_PROMPT = SYSTEM_PROMPT_EN

# Clause/sentence-final punctuation for Chinese/Japanese. Checked in this order
# so a sentence-final mark wins over a clause mark if both are present at the
# same position (they never are, but keep intent explicit).
# NOTE (found via live testing 2026-10-01): the ASCII period was missing here,
# so English/Korean answers (which end sentences with "." rather than the
# full-width "。") never flushed at sentence boundaries, costing ~1-2.5s of
# extra time-to-first-chunk versus Chinese/Japanese. Added "." below.
# Known residual limitation (not handled): a bare "." also matches decimals and
# abbreviations ("Mr.", "1.5") and would cause a premature flush there; the
# arXiv voice-pipeline tutorial referenced earlier guards against this
# explicitly. Not worth the complexity for this prototype's short answers, but
# flag before reusing this chunker on longer or numeric-heavy text.
CLAUSE_BOUNDARIES = "。！？!?、，,."
MIN_CHUNK_CHARS = 8  # guard against flushing near-empty fragments (see module docstring)


def tts_send(chunk: str, t_ready: float, t_start: float):
    """Stub: replace with a real streaming TTS call (e.g. CosyVoice3 bi-streaming API).

    Prints what would be sent, and how long after the request started it became
    available, so this doubles as a latency probe until a real TTS engine is wired in.
    """
    print(f"  [t=+{t_ready - t_start:5.3f}s] -> TTS: {chunk!r}")


def check_grounded_async(question: str, answer: str, rdb_markdown: str):
    """Stub for the separate, non-blocking verification call.

    In production this would be queued (e.g. to a worker, or sampled at some rate)
    and never awaited on the visitor's turn. Left unimplemented here; see
    docs/llm_prompt_design.md section on async verification (still an open TODO there).
    """
    raise NotImplementedError("async verification is designed but not yet implemented")


def stream_answer(base_url: str, model: str, question: str, system_prompt: str = SYSTEM_PROMPT):
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
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    buffer = ""
    full_answer = ""
    chunk_count = 0
    first_chunk_t = None

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content") or ""
            if not content:
                continue

            buffer += content
            full_answer += content

            # Flush at the first clause boundary once the buffer is long enough.
            while True:
                cut = None
                for i, ch in enumerate(buffer):
                    if ch in CLAUSE_BOUNDARIES and i + 1 >= MIN_CHUNK_CHARS:
                        cut = i + 1
                        break
                if cut is None:
                    break
                chunk, buffer = buffer[:cut], buffer[cut:]
                t_ready = time.perf_counter()
                if first_chunk_t is None:
                    first_chunk_t = t_ready
                chunk_count += 1
                tts_send(chunk, t_ready, t_start)

    # Flush whatever is left when the stream ends (per the standard pattern:
    # always flush the tail, even if it never hit a boundary).
    if buffer.strip():
        t_ready = time.perf_counter()
        if first_chunk_t is None:
            first_chunk_t = t_ready
        chunk_count += 1
        tts_send(buffer, t_ready, t_start)

    t_end = time.perf_counter()
    return {
        "question": question,
        "answer": full_answer,
        "chunk_count": chunk_count,
        "time_to_first_chunk_s": (first_chunk_t - t_start) if first_chunk_t else None,
        "total_time_s": t_end - t_start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="この映画が初めて上映されたのはいつですか？")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="unsloth/gemma-4-12b-it-NVFP4")
    ap.add_argument("--lang", choices=["zh", "en"], default="en", help="system prompt (rules) language; exhibit content stays Chinese either way")
    ap.add_argument("--movie", default=None,
                    help="Poster ID (movie_slug or id): load exhibit content from Postgres "
                         "(run via `uv run python scripts/stream_to_tts.py`). "
                         "Default: the built-in L'Arrivée d'un train sample")
    ap.add_argument("--print-prompt", action="store_true", help="print the system prompt and exit")
    args = ap.parse_args()

    if args.movie:
        from vr_guide.content_db import load_exhibit_markdown
        rdb_markdown = load_exhibit_markdown(args.movie)
    else:
        rdb_markdown = RDB_MARKDOWN
    system_prompt = (system_prompt_en if args.lang == "en" else system_prompt_zh)(rdb_markdown)
    if args.print_prompt:
        print(system_prompt)
        return
    print(f"質問: {args.question}  [prompt_lang={args.lang}, movie={args.movie or 'builtin-sample'}]\n")
    r = stream_answer(args.base_url, args.model, args.question, system_prompt)
    print(f"\n完全な回答: {r['answer']!r}")
    print(f"チャンク数: {r['chunk_count']}")
    print(f"最初のチャンクまで: {r['time_to_first_chunk_s']:.3f}s  (= TTS側の体感開始時間)")
    print(f"全体の完了まで: {r['total_time_s']:.3f}s")


if __name__ == "__main__":
    main()
