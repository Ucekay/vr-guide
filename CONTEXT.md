# vr-guide

VRChat映画史博物館ワールド向けの、来場者が見ているポスターを画像照合で同定し、
そのポスターに紐づく解説をAIガイドエージェントに喋らせるシステム。

## Language

**Poster ID**:
画像認識が返す識別子。**`movie_id`（または `movie_slug`）そのもの**であると定義する。1映画1ポスターの原則により、複数アングル・複数スケールで撮った参照写真がすべて同じ1つのポスター＝同じ1つの映画を指すことは、画像認識システムの内部で解決済みとする。この解決ロジックは画像認識システムの内部実装の関心事であり、コンテンツDB（Postgres）側は関与しない。既存の画像認識パイプライン（`match_fast.py`等）の実装は今回の設計の制約にしない。
_Avoid_: id（曖昧）, Reference Image / 参照画像（画像認識システム内部の話であり、このCONTEXTの外）

**Movie（映画）**:
1つの作品。タイトル（中国語・原題）、公開年、所属する Chapter、そして1本の長文解説（Essay）を持つ。ポスターは基本的に1映画につき1枚（1:1）。`film-impression-museum-exhibition-plan.md` の見出しブロック（例:「《火车进站》（... 1895）」）が対応する。
_Avoid_: film, work, title, Panel（廃案。画像単位で解説を分割する旧設計は採用しなかった）

**Essay（解説文）**:
`film-impression-museum-exhibition-plan.md` に書かれた、1つの Movie につき1本の長文プレーンテキスト。監督・国・尺などの構造化メタ情報は含まず、地の文の中に言及がある程度。
_Avoid_: description

**Chapter / Era（章/時代）**:
映画史を7つに区分した時代単位（era1〜era7）。VRChatワールド内の展示エリア区分でもある。1つの Movie は1つの Chapter に属する。
_Avoid_: section, category

## 確定した設計方針

- **コンテンツの正本（執筆）**: `film-impression-museum-exhibition-plan.md`（Markdown）で執筆を続ける。`data/refs/eraN/content.md` は本システムでは一切参照しない（無関係・考慮外）。
- **コンテンツの正本（保存先）**: 上記Markdownの内容を取り込んだ **Postgres** が正本になる。取り込みスクリプトでMarkdown→Postgresに同期する。
- **スキーマ**（確定）:
  ```
  chapters (id PK, era_no, title_zh, year_from, year_to)
  movies   (id PK, movie_slug UNIQUE, chapter_id FK, title_zh, title_orig, year, essay_md)
  ```
  Poster ID → Movie のマッピング専用テーブルは持たない（Poster ID自体が movie の PK/slug であるため不要）。
- **idを受け取った後の取得フロー**: 受け取った id（＝`movie_id`/`movie_slug`）で `movies` を1行取得し `chapters` をJOIN（1クエリ）→ `chapter.title_zh` / `movie.title_zh` / `movie.title_orig` / `movie.year` / `movie.essay_md` をMarkdownテンプレートに組み立ててAIエージェント（LLM）のコンテキストとして渡す。
- **エージェント応答方式**: 固定文の読み上げではなく、上記Markdownをコンテキストとして渡し、来場者の自由質問にLLMが答える方式（自由対話型）。
- **エージェント応答範囲**: 1回の会話は「今認識している Poster ID が属する Movie」の情報に閉じる。他の Movie を横断した質問には（今回のスコープでは）対応しない。
- **複数ポスター同時検出**: 実運用上は基本的に1スクリーンショットにつき Poster ID は1つしか返らない想定。
- **実行形態**: 常時稼働のサーバー1つに複数来場者からのリクエストが集まるサーバー集中型。DBもサーバー側で共有（Postgres）。詳細は [[0001-postgres-server-content-store]] 参照。
- **多言語対応**: 現状は中国語決め打ち。多言語化は将来の別ADRとして先送り（YAGNI）。

## LLMモジュール（説明生成）の確定方針

- **Grounded Answering（根拠限定回答）の原則**: LLMは `movies.essay_md` に書かれている内容だけを根拠に回答し、そこに書かれていない事柄を聞かれたら一般知識で補完せず「ここでは分かりません」の意の応答をする。画像認識パイプラインで「ポスターが無い画面で誤って何かを答えるのは最悪の失敗」としてprecisionを最優先した設計判断（[[report_2026-08-19]]）と一貫させる。
  _Avoid_: 補完回答、推測回答
- **話題逸脱時の定型応答**: (i) 別の映画について聞かれたら、今の展示に注意を戻す誘導をする。(ii) 映画と無関係な話題を振られたら、ガイドとしての役割を明示して穏やかに断る。
- **出力モダリティ**: 音声合成（TTS）で読み上げられる前提。Markdown記法（太字・箇条書き等）を使わず、短い文に区切った自然な話し言葉で出力する。
- **会話履歴（Conversation History）**: 同じ Poster ID を見ている間だけ保持し、Poster ID が変わったらリセットする（＝「今見ている映画に閉じる」という応答範囲の確定方針と一致させる）。履歴の保持責務はLLMモジュールの外（呼び出し側のサーバー）にあり、LLMモジュール自体はステートレスな関数として設計する：`(質問, RDBデータ, 会話履歴) → 応答テキスト`。
- **応答言語**: 来場者の質問と同じ言語に翻訳して答える（`essay_md`自体は中国語のみだが、回答時に訳す）。
- **推論モデル**: ローカルLLM、Qwen2.5-Instruct系（7B〜14B程度）を想定。推論ハードウェアは画像認識パイプラインと同じ NVIDIA GB10（CUDA 13, aarch64）を流用する。選定理由: `essay_md`が中国語であり、かつ来場者の質問言語への翻訳が要るため、中国語・多言語性能に強いモデルファミリーを優先した。
  _Avoid_: クラウドAPI（Claude/OpenAI等）は今回は不採用
- **出力契約（構造化出力）**: LLMモジュールの出力は `{"answer": string, "grounded": boolean, "quote": string}` のJSON。`answer`（読み上げ用テキスト）だけをTTSに渡し、`grounded`/`quote`（essay_mdからの引用根拠）はログに残して、Grounded Answeringの原則が守られているかを後から検証できるようにする。画像認識パイプラインの `score.py` と同様の「実測して検証する」文化をLLM側にも適用する。
