# vr-guide

VR映画ポスター展示 × ガイドエージェント。
VRChatスクリーンショット内のポスターを、参照ポスターDBに対して
instance-level で同定するパイプライン。

## 構成

```
src/vr_guide/
  lgtools.py     共通ユーティリティ（特徴抽出 / 矩形検出 / 投票bank / mnn+RANSAC照合）
  build_db.py    参照ポスター群 -> マルチスケール特徴DB (.pt)
  match_fast.py  スクショ -> 矩形提案 -> クロップ拡大 -> 投票検索 -> 幾何検証
  score.py       結果CSV vs 正解ラベル の採点（棄却=none を含む）

data/refs/       参照ポスター 334枚 (era1..era7)
data/queries/    VRChatスクショ 39枚 (438100_*.jpg, era1のみ)
data/labels_era1.csv  正解ラベル（none行=ポスター無しの棄却評価つき）
db/fast.pt       構築済み特徴DB（SuperPoint, scales=[1024,700,480,320,220], 1635 entries）
out/             結果CSV・可視化
```

`data/` `db/` `out/` は容量が大きいため git 管理外。

## セットアップ

```bash
uv sync
```

GB10 (sm_121) / CUDA 13 環境向けに torch は `download.pytorch.org/whl/cu130` から
取得する設定（`[tool.uv.sources]`）。LightGlue は rev `eb42fee` に固定して git から取得
（使用しているのは SuperPoint 等の extractor のみ。matcher は自前の mutual-NN）。

## 実行

```bash
# 照合（構築済みDBを使う場合）
uv run match --db db/fast.pt --query data/queries --out-csv out/repro.csv --viz-dir out/viz --bench

# 採点
uv run score --results out/repro.csv --labels data/labels_era1.csv --lang en

# DB再構築
uv run build-db --ref-dir data/refs --out db/fast.pt --scales 1024 700 480 320 220
```

## 映画解説DB（Postgres）

`film-impression-museum-exhibition-plan.md` を正本として、Postgres の `chapters` / `movies` に同期する
（設計: `CONTEXT.md`, `docs/adr/0001-postgres-server-content-store.md`）。

```bash
docker compose up -d --wait          # Postgres 17 (127.0.0.1:5432, vrguide/vrguide)
uv run sync-content --dry-run        # パース結果の確認のみ
uv run sync-content                  # Markdown -> Postgres（movie_slug で upsert、消えた映画は削除）
uv run get-movie the-godfather       # Poster ID（movie_slug または id）で1件取得
```

接続先は `$DATABASE_URL`（既定 `postgresql://vrguide:vrguide@localhost:5432/vrguide`）。

## ベースライン実測値 (2026-08-19, NVIDIA GB10 / CUDA 13.0 / torch 2.13.0+cu130)

era1 テストセット 39枚:

| 指標 | 値 |
|---|---|
| Fully correct | 36/39 (92.3%) |
| precision | **1.000** (FP=0) |
| recall | 0.923 |
| F1 | 0.960 |
| 速度 | **126 ms/query (7.9 fps)** |

Windows 上で取得済みの `out/fast.csv` と**検出集合が完全一致**（移植の妥当性を確認済み）。

段階別内訳 (ms/query): extract 78.5 / retrieve 21.6 / propose 6.8 / read 6.4 / verify 6.1 / prep 1.1

### 取りこぼし3件の内訳（誤認識ゼロ）

| shot | 内容 |
|---|---|
| `438100_50` | 2枚写っているうち1枚のみ検出（`factory_gate_02` は取れ、`_01` を落とす） |
| `438100_61` | 同上（`train_robbery_03` は取れ、`_02` を落とす） |
| `438100_51` | 正解 `factory_gate_02` が rank-1 だが inliers=9 で閾値 `--min-inliers 10` に1足りず棄却 |

- 正解が rank 外に落ちた例は無い。**検索段(Stage 1)の天井は 39/39。**
- ポスター無しの2枚 (`438100_54`, `438100_70`) はいずれも正しく棄却 → FP=0。
- → 改善余地は「同一フレーム内の複数ポスター検出」と「閾値校正」であって、
  記述子や照合手法の差し替えではない。

## TODO

- [ ] era2..era7 のスクショとラベルを用意し、DB 334枚全域で評価（現状 era1 のみ）
- [ ] `438100_50` / `_61` の2枚目落ちを追う（`--max-props` / `--iou-thresh` / `--budget`）
- [ ] `--min-inliers` を precision を保ったまま下げられるか、閾値スイープで校正
- [ ] 合成クエリ生成（既知ホモグラフィ + ダウンサンプル + ブラー + 遮蔽）で
      視野角 × 実効解像度 × 遮蔽率の条件別 breakdown を取る
- [ ] raycast ベースラインとの比較
