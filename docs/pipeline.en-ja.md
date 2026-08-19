# VRChat Poster Recognition / VRChat ポスター識別

**EN** — Given a VRChat screenshot, identify **which reference image** from the
museum library it shows, and localise it in the frame.

**JA** — VRChat のスクリーンショットを入力すると、それが美術館ライブラリ内の
**どの参照画像なのか**を特定し、フレーム内での位置を矩形で示します。

| | Result / 結果 |
| --- | --- |
| Speed / 速度 | **~0.17 s per frame** (158–180 ms measured, RTX 5070 Laptop) |
| Accuracy / 精度 | **36 / 39** fully correct · precision **1.000** · recall **0.923** · F1 **0.960** |
| Library size / ライブラリ規模 | 327 reference images → 1635 multi-scale entries |

**EN** — Of the 39 test screenshots, 2 (`438100_54`, `438100_70`) contain no poster
at all — only a 3D shell-capsule prop — so returning "none" is the correct
behaviour there. Effective accuracy is therefore **36 / 37 = 97%**. The remaining
miss, plus the second poster in two other frames, are all ~75 px thumbnails.

**JA** — テスト画像 39 枚のうち 2 枚（`438100_54`、`438100_70`）にはポスターが一切写って
おらず、3D の砲弾カプセルのみです。したがって "none" を返すのが正解であり、
実質的な精度は **36 / 37 = 97%** です。残る 1 枚の見逃しと、他 2 枚における 2 枚目の
ポスターは、いずれも約 75 px の極小サムネイルです。

---

## 1. How it works / 仕組み

### 1.1 The core difficulty / 本質的な難しさ

**EN** — The hard part is not "do these two images look alike" — it is **scale**.
A poster occupies only 80–250 px inside a 1920 px screenshot, while its reference
image is 1000+ px: a **3–8× gap**. On top of that come perspective distortion
(you photograph the wall at an angle), cast shadows, and unlit night rooms.

**JA** — 難しいのは「2 枚の画像が似ているか」ではなく **スケール差**です。
1920 px のスクリーンショット内でポスターが占めるのはわずか 80〜250 px なのに対し、
参照画像は 1000 px 以上、つまり **3〜8 倍の差**があります。さらに射影変形（壁を斜めから
撮影するため）、落ち影、夜間の暗い部屋といった条件が重なります。

### 1.2 Pipeline / パイプライン

```
① Read frame + global CLAHE          13 ms   only to help step ②
② Find "picture-frame" quadrilaterals 14 ms   typically 4–5, plus a whole-frame fallback
③ Per quad: crop → local CLAHE → zoom to 1024 px → extract features   95 ms
④ Per quad: vote against the descriptor bank                          29 ms
⑤ Verify the strongest candidates under a global budget               14 ms
⑥ Rank by inliers + suppress overlapping detections                   15 ms
                                                                    ───────
                                                                     180 ms
```

| Step | English | 日本語 |
| --- | --- | --- |
| ① | Read the image and apply CLAHE to the whole frame. Used **only** for edge detection in step ②; cropping in step ③ reads the *original* frame. | 画像を読み込み、全体に CLAHE を適用。これは②のエッジ検出**専用**で、③の切り出しは*元画像*から行います。 |
| ② | Canny edges → contours → 4-corner convex polygons. Class-agnostic: it knows nothing about posters, only about quadrilaterals. A whole-frame candidate is always appended, covering posters that fill the view or rooms too dark to find edges in. | Canny → 輪郭抽出 → 4 頂点の凸多角形。ポスターという概念は持たず、四角形だけを探します。画面全体を常に候補に追加し、ポスターが画面いっぱいの場合や暗すぎてエッジが出ない場合に備えます。 |
| ③ | Crop with 20% margin, apply CLAHE **to the crop**, upscale so the long edge is 1024 px, then run SuperPoint. | 20% の余白を付けて切り出し、**切り出した領域に**CLAHE を適用、長辺 1024 px まで拡大してから SuperPoint を実行。 |
| ④ | Each query keypoint finds its global nearest neighbour in one flat descriptor bank and votes for the entry that owns it. | 各キーポイントが、平坦化された記述子バンク全体から最近傍を探し、その所属エントリに投票します。 |
| ⑤ | Candidates from *all* quads are queued by vote count and verified with mutual-NN + Lowe ratio + RANSAC until the budget runs out. | *全*四角形の候補を得票順に一列に並べ、予算が尽きるまで相互最近傍 + Lowe 比率検定 + RANSAC で検証します。 |
| ⑥ | Sort by RANSAC inlier count; drop anything below threshold; suppress detections overlapping a higher-scoring one. | RANSAC のインライア数で並べ替え、閾値未満を除外し、上位と重なる検出を抑制します。 |

### 1.3 The four design decisions that matter / 重要な 4 つの設計判断

**EN** — All four were arrived at by measurement, not intuition.

**JA** — 4 つとも直感ではなく実測から得られたものです。

**① CLAHE is applied *after* cropping, not to the whole frame.**
A poster washed out by a light gradient occupies ~2% of the frame, so global
normalisation averages its problem away. Once cropped, it *is* the whole image
and contrast recovery works. Three previously-unfindable screenshots jumped to
258–522 inliers.

> **JA** — CLAHE は全体ではなく**切り出し後**に適用します。光のグラデーションで白飛び
> したポスターはフレームの約 2% しか占めないため、全体正規化では問題が平均化されて
> しまいます。切り出せばそれが画像全体になるので、コントラスト復元が機能します。
> 従来まったく検出できなかった 3 枚が 258〜522 インライアまで改善しました。

**② The reference library stores 5 scales per image.**
Descriptors are scale-sensitive. Measured on one screenshot: with a
**single-scale** bank the correct answer ranked **28th** (top vote was 3 — pure
noise); with a **five-scale** bank it ranked **1st with 34 votes** against 4 for
the runner-up.

> **JA** — 参照ライブラリは 1 枚につき 5 段階のスケールを保持します。記述子はスケールに
> 敏感です。あるスクリーンショットでの実測では、**単一スケール**のバンクだと正解は
> **28 位**（最高得票が 3 票＝ノイズと同然）でしたが、**5 スケール**では
> **1 位・34 票**（2 位は 4 票）でした。

**③ Retrieval is one fused GEMM.**
Comparing the query against each entry separately means one tiny matrix multiply
per entry, driven from Python: **706 ms** for 1635 entries. Stacking all
descriptors into one matrix and doing a single large multiply performs the same
arithmetic in **20 ms**. The voting bank can also stay thin — only the strongest
256 descriptors per entry — because voting only needs to rank, not to verify.

> **JA** — 検索は 1 回の融合 GEMM で行います。エントリごとに個別比較すると、Python から
> 1635 回の小さな行列積を呼ぶことになり **706 ms** かかります。全記述子を 1 つの行列に
> 積み上げて 1 回の大きな行列積にすると、同じ演算量が **20 ms** で済みます。投票用の
> バンクは各エントリ上位 256 記述子だけで十分です（投票は順位付けのためであり、
> 検証は別途行うため）。

**④ The verification budget is global, not per quad.**
With a per-quad quota, four blank-wall proposals burn 16 matcher calls. Pooling
all candidates and spending highest-vote-first means blank walls cost almost
nothing, while a faint-but-correct vote on a small thumbnail still gets its
chance. This took verification from **174 ms to 14 ms** *and* improved accuracy.

> **JA** — 検証予算は四角形ごとではなく全体で管理します。四角形ごとの割り当てだと、
> 空白の壁 4 つで 16 回の照合を浪費します。全候補をまとめて得票順に消費すれば、
> 空白の壁はほぼコストゼロになり、小さなサムネイル上の「得票は低いが正しい」候補にも
> 機会が残ります。これにより検証は **174 ms → 14 ms** になり、精度も向上しました。

### 1.4 Which parts are neural networks? / どこがニューラルネットワークか

**EN** — Exactly one: **SuperPoint**, the keypoint detector and descriptor.
Everything else is classical. OpenCV here supplies classical algorithm
implementations, not any deep model.

**JA** — ニューラルネットワークは **SuperPoint**（特徴点検出＋記述子）**ただ 1 つ**です。
それ以外はすべて古典的アルゴリズムです。ここでの OpenCV は古典的手法の実装を提供して
いるだけで、深層学習モデルは含みません。

| Stage / 段階 | Method / 手法 | NN? | Source / 出典 |
| --- | --- | --- | --- |
| Keypoints + descriptors | **SuperPoint** | **Yes** | DeTone / Malisiewicz / Rabinovich, CVPRW, [arXiv:1712.07629](https://arxiv.org/abs/1712.07629) (Magic Leap). Code and weights ship with **[cvg/LightGlue](https://github.com/cvg/LightGlue)** |
| Quad proposals | Canny → contours → approxPolyDP | No | Canny 1986; Douglas–Peucker 1973 (OpenCV) |
| Contrast | CLAHE | No | Zuiderveld, *Graphics Gems IV*, 1994 (OpenCV) |
| Retrieval | Nearest-neighbour voting (one GEMM) | No | Written for this project; the idea is local-feature voting, in the lineage of Sivic & Zisserman, *Video Google*, ICCV 2003 |
| Matching | Mutual-NN + Lowe ratio test | No | Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004 |
| Geometric check | RANSAC (USAC_MAGSAC) + homography | No | Fischler & Bolles 1981; MAGSAC++: Barath et al., CVPR 2020 (OpenCV) |

**EN** — The standard three-stage pipeline in this field is **① detect + describe
→ ② match → ③ geometrically verify**. This project uses a classical method for
stage ②, but stage ① is unchanged — so it is *not* "plain classical CV": the
representation that carries the whole system is still learned. Swapping stage ①
for classical SIFT was measured: 35/39 correct with 1 false positive, and 2.2×
slower (OpenCV SIFT is CPU-only).

**JA** — この分野の標準的な 3 段構成は **① 検出＋記述 → ② 対応付け → ③ 幾何検証** です。
本プロジェクトは②に古典的手法を使っていますが、①は変更していません。つまり
「純粋な古典的 CV」ではなく、システム全体を支える表現は依然として学習済みモデルによる
ものです。①を古典的な SIFT に置き換えた実測では、正解 35/39・誤検出 1 件・速度 2.2 倍遅
（OpenCV の SIFT は CPU 専用）という結果でした。

### 1.5 Why the inlier count is the confidence score / なぜインライア数が信頼度なのか

**EN** — Counting "how many keypoints look similar" is easy to fool: two
unrelated images can produce a dozen coincidental matches. RANSAC additionally
demands that the matches be **explainable by a single perspective transform** —
posters are planar, so this holds for a true match and essentially never for
coincidences. The surviving count is the score.

**JA** — 「似ている特徴点がいくつあるか」だけでは簡単に騙されます。無関係な 2 枚でも
偶然の対応が十数個生じるからです。RANSAC はさらに、それらの対応が
**単一の射影変換で説明できること**を要求します。ポスターは平面なので正しい対応なら
成立し、偶然の一致ではまず成立しません。生き残った対応の数がスコアになります。

| | Inliers / インライア数 |
| --- | --- |
| True matches / 正しい一致 | 12 – 592 |
| Frames containing no poster / ポスターが無いフレーム | ≤ 6 |
| Default `--min-inliers` | 10 |

**EN** — Note that the inlier **ratio** does *not* separate true from false:
RANSAC can find a self-consistent subset even inside noise, so false matches also
reach ratios of 0.6–0.9. Only the absolute count separates cleanly.

**JA** — なお、インライア**率**では真偽を区別できません。RANSAC はノイズの中からでも
自己無矛盾な部分集合を見つけられるため、誤対応でも 0.6〜0.9 に達します。
きれいに分離できるのは絶対数だけです。

---

## 2. Setup / セットアップ

```powershell
conda create -n lightglue python=3.11 -y
conda activate lightglue

# RTX 50-series (Blackwell, sm_120) requires cu128 or newer wheels.
# RTX 50 系（Blackwell, sm_120）には cu128 以降のホイールが必須です。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "kornia>=0.6.11" opencv-python matplotlib tqdm

git clone https://github.com/cvg/LightGlue.git E:\poster\LightGlue
pip install -e E:\poster\LightGlue --no-deps      # --no-deps: do not downgrade torch

python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
# -> 2.11.0+cu128 True
```

> ⚠️ **EN — Do not delete `E:\poster\LightGlue`.** LightGlue is no longer used for
> matching, but **SuperPoint's code and weights live in that repository** and the
> package is installed in editable mode pointing at it. Removing it causes
> `ModuleNotFoundError: No module named 'lightglue'`. SuperPoint weights
> (~50 MB) download automatically on first run.
>
> ⚠️ **JA — `E:\poster\LightGlue` を削除しないでください。** 照合には LightGlue を使って
> いませんが、**SuperPoint のコードと重みはこのリポジトリに含まれ**、パッケージは
> editable モードでこのパスを参照しています。削除すると
> `ModuleNotFoundError: No module named 'lightglue'` になります。
> SuperPoint の重み（約 50 MB）は初回実行時に自動ダウンロードされます。

---

## 3. Usage / 使い方

### Step 1 — Build the feature database (once) / 特徴データベースの構築（1 回のみ）

**EN** — Rebuild whenever reference images are added, removed or changed.
Takes ~30 s.
**JA** — 参照画像を追加・削除・変更したら必ず再構築してください。約 30 秒です。

```powershell
python build_db.py --ref-dir "E:\poster\VRChat电影馆" --out db\fast.pt `
                   --scales 1024 700 480 320 220 --max-kp 1024
```

```
[db] 327 reference images x 5 scales -> 1635 entries
extract: 100%|██████████████████████████| 327/327 [00:30<00:00, ...]
[db] saved db\fast.pt  (1635 entries, 1082706 keypoints, 560 MB)
```

### Step 2 — Recognise / 識別

```powershell
# A whole folder / フォルダ全体
python match_fast.py --db db\fast.pt --query "E:\poster\Test_Set" `
                     --out-csv out\fast.csv --viz-dir out\viz_fast

# A single image, with per-stage timing / 単一画像＋段階別計測
python match_fast.py --db db\fast.pt --query "E:\poster\Test_Set\438100_38.jpg" --bench
```

### Step 3 — Score against an answer key (optional) / 正解データによる採点（任意）

```powershell
python score.py --results out\fast.csv --labels labels_example.csv --lang en
```

**EN** — `--lang en` switches the report to English; omit it for Chinese.
**JA** — `--lang en` で英語表示に切り替わります。省略すると中国語表示です。

### Key options / 主なオプション

| Option | Default | English | 日本語 |
| --- | --- | --- | --- |
| `--budget` | 32 | Max verification calls per frame, spent highest-vote first. Cheap — raising it barely costs time. | 1 フレームあたりの検証回数上限。得票順に消費。安価なので増やしても時間はほぼ増えません。 |
| `--max-props` | 4 | Number of quad proposals kept (largest first). Raise when a frame holds many pictures. | 保持する四角形候補の数（面積順）。1 画面に多数の絵がある場合に増やします。 |
| `--crop-target` | 1024 | Long edge each crop is zoomed to. **Do not lower** — costs recall, saves almost nothing. | 各切り出しを拡大する長辺。**下げないでください**。再現率が落ち、時間はほぼ短縮されません。 |
| `--min-inliers` | 10 | Detection threshold. | 検出とみなす閾値。 |
| `--vote-kp` / `--bank-kp` | 512 / 256 | Query keypoints used for voting / descriptors per entry in the vote bank. | 投票に使うクエリ特徴点数 / 投票バンク内の 1 エントリあたり記述子数。 |
| `--ratio` | 0.90 | Lowe ratio threshold; lower is stricter. | Lowe 比率検定の閾値。小さいほど厳格。 |
| `--fp32` | — | Disable half-precision extraction. | 半精度での特徴抽出を無効化。 |
| `--bench` | — | Print per-stage timing. **Run this before tuning anything.** | 段階別の所要時間を表示。**調整前にまず実行してください。** |
| `--device` | cuda | Use `cpu` if no GPU (much slower). | GPU が無い場合は `cpu`（大幅に低速）。 |

---

## 4. Input and output / 入力と出力

### 4.1 Input / 入力

**EN** — Two things: a folder of **reference images** (recursively scanned;
`.jpg .png .webp .bmp .tif`), and one or more **query screenshots**.

**JA** — 必要なものは 2 つ。**参照画像**のフォルダ（再帰的に走査。
`.jpg .png .webp .bmp .tif`）と、**クエリとなるスクリーンショット**です。

```
E:\poster\
├─ VRChat电影馆\          ← reference library / 参照ライブラリ
│   ├─ era1\
│   │   ├─ train_arrival_01.jpg
│   │   ├─ factory_gate_01.jpg
│   │   └─ ...
│   └─ era2\ ... era7\
└─ Test_Set\              ← query screenshots / クエリ画像
    ├─ 438100_38.jpg      (1920 x 969)
    └─ ...
```

**EN** — Result names are paths **relative to `--ref-dir`**, e.g.
`era1/train_arrival_03.jpg`, so subfolder names show up in every output — name
them meaningfully.

**JA** — 出力される名前は **`--ref-dir` からの相対パス**（例：
`era1/train_arrival_03.jpg`）です。サブフォルダ名がすべての出力に現れるため、
意味のある名前を付けてください。

### 4.2 Output 1 — console / コンソール

```
[db] 1635 entries / 397766 descriptors (204 MB), scales=[1024, 700, 480, 320, 220]
[  1/39] 438100_38.jpg         244ms props=3  -> era1/train_arrival_03.jpg(350)
[  2/39] 438100_39.jpg         118ms props=3  -> era1/train_arrival_03.jpg(288)
[  3/39] 438100_40.jpg         165ms props=5  -> era1/train_arrival_03.jpg(437)
[  4/39] 438100_41.jpg          88ms props=2  -> era1/train_arrival_01.jpg(235)
...
[ 17/39] 438100_54.jpg         208ms props=5  -> -- none --
...
[ 39/39] 438100_76.jpg         216ms props=5  -> era1/train_arrival_03.jpg(592)
[out] out\fast.csv
[done] 36/39 matched | mean 179 ms/query (5.6 fps)
```

| Field | English | 日本語 |
| --- | --- | --- |
| `244ms` | Wall-clock time for this frame. | このフレームの処理時間。 |
| `props=3` | Quad proposals found (including the whole-frame fallback). | 検出された四角形候補の数（画面全体のフォールバックを含む）。 |
| `era1/train_arrival_03.jpg` | The identified reference image. | 特定された参照画像。 |
| `(350)` | RANSAC inliers = confidence. ≥ 60 is labelled `high`. | RANSAC のインライア数＝信頼度。60 以上は `high` と表示。 |
| `-- none --` | No poster found. Correct when the frame genuinely has none. | ポスターが見つからず。実際に写っていない場合はこれが正解です。 |

**EN** — If a frame shows several posters, each is listed, comma-separated, and
written as its own CSV row.
**JA** — 1 フレームに複数のポスターが写っている場合は、カンマ区切りで併記され、
CSV にもそれぞれ 1 行として出力されます。

### 4.3 Output 2 — `--bench` timing / 計測出力

```
[bench] per-query average
    read         13.0 ms
    propose      13.9 ms
    prep          2.7 ms
    extract      95.1 ms      <- dominant cost / 支配的なコスト
    retrieve     29.2 ms
    verify       13.6 ms
    other        12.3 ms
```

### 4.4 Output 3 — CSV / CSV 出力

**EN** — Top-5 candidates per screenshot (`--topk`). Real rows from this test set:
**JA** — スクリーンショットごとの上位 5 候補（`--topk`）。本テストセットの実データ：

```csv
query,rank,ref,inliers,matches,confidence,ref_scale,detected
438100_38.jpg,1,era1/train_arrival_03.jpg,350,360,high,480,1
438100_38.jpg,2,era3/citizen_kane_susan_01.jpg,4,16,low,1024,0
438100_50.jpg,1,era1/factory_gate_02.jpg,91,126,high,480,1
438100_54.jpg,1,era3/wizard_of_oz_kansas_01.jpg,5,34,low,700,0
438100_54.jpg,2,era2/caligari_02.png,5,25,low,700,0
```

| Column | English | 日本語 |
| --- | --- | --- |
| `query` | Screenshot filename. | スクリーンショットのファイル名。 |
| `rank` | 1 = best guess. Lower ranks are kept for inspection, not asserted. | 1 が最有力。下位は確認用に残しているだけで、主張ではありません。 |
| `ref` | Candidate reference image, relative to `--ref-dir`. | 候補となる参照画像（`--ref-dir` からの相対パス）。 |
| `inliers` | RANSAC inliers — the score. | RANSAC のインライア数＝スコア。 |
| `matches` | Correspondences before geometric verification. | 幾何検証前の対応数。 |
| `confidence` | `high` ≥ 60, `medium` ≥ 10, else `low`. | `high` は 60 以上、`medium` は 10 以上、それ以外は `low`。 |
| `ref_scale` | Which pyramid scale won — useful for diagnosing scale gaps. | どのスケールが選ばれたか。スケール差の診断に有用。 |
| `detected` | 1 = reported as a detection; 0 = candidate only. | 1 は検出として報告、0 は候補止まり。 |

**EN** — Note `438100_54`: every candidate scores ≤ 5 and `detected` is 0 for all
of them. That frame contains no poster, and the pipeline correctly declines to
guess. The gap between 5 and the 10-inlier threshold is what makes that decision
safe.

**JA** — `438100_54` に注目してください。全候補のスコアが 5 以下で、`detected` はすべて
0 です。このフレームにはポスターが無く、システムは正しく判断を保留しています。
5 と閾値 10 の間の差が、この判断を安全にしています。

### 4.5 Output 4 — annotated images / 注釈付き画像

**EN** — With `--viz-dir`, every detection is drawn on the original screenshot:
the reference image's four corners projected through the estimated homography,
plus the filename and inlier count. This is the fastest way to sanity-check a new
dataset — you can see at a glance whether the box landed on the right picture.

**JA** — `--viz-dir` を指定すると、検出結果が元のスクリーンショット上に描画されます。
推定したホモグラフィで参照画像の 4 隅を射影した矩形に、ファイル名とインライア数を
併記します。新しいデータセットの妥当性確認にはこれが最速で、矩形が正しい絵に
乗っているかひと目で判断できます。

```
out/viz_fast/
├─ 438100_38.jpg     ← green quad labelled "era1/train_arrival_03.jpg (350)"
├─ 438100_39.jpg
└─ ...               (36 files — only frames with a detection)
```

### 4.6 Output 5 — score report / 採点レポート

**EN** — Provide an answer key as a two-column CSV. Use `none` for a screenshot
containing no poster; repeat a filename to list several posters.

**JA** — 正解データは 2 列の CSV で用意します。ポスターが無いスクリーンショットには
`none` を、複数のポスターがある場合は同じファイル名を複数行に記述します。

```csv
query,ref
438100_38.jpg,era1/train_arrival_03.jpg
438100_50.jpg,era1/factory_gate_01.jpg
438100_50.jpg,era1/factory_gate_02.jpg
438100_54.jpg,none
```

```
$ python score.py --results out\fast.csv --labels labels_example.csv --lang en

Screenshots      39
Fully correct    36/39  (92.3%)
Hit / FP / Miss  36 / 0 / 3
precision        1.000   (how many reported answers are right)
recall           0.923   (how many of the expected ones were found)
F1               0.960

3 with problems:
  438100_50.jpg         missing era1/factory_gate_01.jpg
  438100_51.jpg         missing era1/factory_gate_02.jpg
  438100_61.jpg         missing era1/train_robbery_02.jpg
```

**EN** — Reading the numbers: low **precision** means it guesses wrong too often —
raise `--min-inliers`. Low **recall** means it misses too often — lower
`--min-inliers`, or raise `--max-props` if frames hold many pictures.

**JA** — 数値の読み方：**precision** が低い＝誤答が多い → `--min-inliers` を上げます。
**recall** が低い＝見逃しが多い → `--min-inliers` を下げるか、1 画面に絵が多い場合は
`--max-props` を上げます。

---

## 5. Known limitations / 既知の限界

**EN**

1. **~75 px thumbnails are missed.** Three detections in this test set are lost
   this way. The bottleneck is retrieval (step ④), not the proposal or budget:
   such a crop yields too few keypoints for the correct entry to enter the
   shortlist. Measured — raising `--max-props 8 --budget 64` does *not* fix it.
   A semi-dense matcher such as
   [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) is the right tool
   for this class of target.
2. **Near-duplicate references co-occur.** `train_arrival_01`/`02` are two stills
   from the same film, and `moon_trip_02` (a poster) *embeds* `moon_trip_01`
   (the still). Both appear in the candidate list, but the correct one always
   scores higher and the overlap suppression in step ⑥ removes the other.
3. **The library contains exact duplicates**, e.g.
   `era4/3_Rashomon_1950_IMDb.png` and `era4/rashomon_02.jpg`. These necessarily
   receive identical scores.
4. **Windows + non-ASCII paths.** `cv2.imread` silently fails on paths containing
   non-ASCII characters, and the reference folder here is named `VRChat电影馆`.
   All IO therefore goes through `cv2.imdecode(np.fromfile(...))` — do not
   substitute `imread` back in.

**JA**

1. **約 75 px のサムネイルは検出できません。** 本テストセットではこれにより 3 件を
   取りこぼしています。ボトルネックは候補抽出でも予算でもなく検索（④）です。
   この大きさの切り出しでは特徴点が少なすぎて、正解が候補リストに入りません。
   `--max-props 8 --budget 64` に上げても改善しないことを実測済みです。この種の対象には
   [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) のような semi-dense 手法が
   適しています。
2. **酷似した参照画像が同時に上位に来ます。** `train_arrival_01`/`02` は同じ映画の
   2 枚のスチルであり、`moon_trip_02`（ポスター）には `moon_trip_01`（スチル）が
   *埋め込まれて*います。両方が候補に挙がりますが、正解が常に高スコアとなり、
   ⑥の重複抑制で他方は除去されます。
3. **ライブラリに完全な重複画像が存在します**（例：`era4/3_Rashomon_1950_IMDb.png` と
   `era4/rashomon_02.jpg`）。これらは必然的に同一スコアになります。
4. **Windows と非 ASCII パス。** `cv2.imread` は非 ASCII を含むパスで無言で失敗します。
   本プロジェクトの参照フォルダ名は `VRChat电影馆` です。そのため入出力はすべて
   `cv2.imdecode(np.fromfile(...))` を経由しています。`imread` に戻さないでください。

---

## 6. Files / ファイル構成

```
poster_match/
├─ README.en-ja.md      This file / 本ファイル
├─ README.md            Full technical notes, measurements, design history (Chinese)
├─ 新手指南.md           Beginner's walkthrough (Chinese)
├─ lgtools.py           Library: IO, feature DB, descriptor bank + voting,
│                       quad proposals, crop & zoom, mutual-NN + RANSAC, drawing
├─ build_db.py          Build the reference feature database
├─ match_fast.py        Recognition entry point
├─ score.py             Score results against an answer key
├─ labels_example.csv   Answer key for Test_Set (manually verified)
├─ db/fast.pt           Feature database (560 MB; 327 images x 5 scales)
└─ out/                 fast.csv + viz_fast/
```

**EN** — `build_db.py` also accepts `--extractor sift | aliked | disk` if you want
to compare front-ends.
**JA** — フロントエンドを比較したい場合、`build_db.py` は
`--extractor sift | aliked | disk` も受け付けます。
