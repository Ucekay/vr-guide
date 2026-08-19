# VRChat 电影馆海报识别

输入一张 VRChat 截图，输出它拍到的是 `VRChat电影馆/` 里的**哪一张**参考图，并框出位置。

* **速度**：约 **0.17 秒/张**（同机多次测量 158–176 ms，RTX 5070 Laptop）
* **准确率**：39 张测试集里 **36 张完全正确**，precision **1.000**、recall **0.923**、F1 **0.960**

39 张里有 2 张（`438100_54`、`438100_70`）画面中只有一个 3D 炮弹模型、根本没有海报，
输出 "none" 是正确行为。所以实际是 **36 / 37 = 97%**。剩下 1 张漏检、以及 `_50`/`_61`
各漏掉的第二张小图，都是 ~75 px 的超小缩略图。

一致性检查：所有命中**全部落在 era1**（候选库共 327 张、覆盖 era1–era7），
且 era1 的 10 张参考图每一张都被命中过。

> 新手向的通俗讲解（架构比喻 + 代码逐步详解 + 怎么换自己的测试集）见 **[新手指南.md](新手指南.md)**。
> 本文件偏技术：参数表、实测数据、设计取舍。

---

## 一、环境部署（已完成，此处供复现/重装）

```powershell
conda create -n lightglue python=3.11 -y
conda activate lightglue

# RTX 50 系（Blackwell, sm_120）必须用 cu128 及以上的轮子，
# 默认 PyPI 的 torch 会报 "no kernel image is available"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "kornia>=0.6.11" opencv-python matplotlib tqdm

git clone https://github.com/cvg/LightGlue.git E:\poster\LightGlue
pip install -e E:\poster\LightGlue --no-deps        # --no-deps 防止它把 torch 降级

python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

本机：`2.11.0+cu128 True NVIDIA GeForce RTX 5070 Laptop GPU`。

> ⚠️ **`E:\poster\LightGlue` 不能删。** 虽然现在不用 LightGlue 做匹配了，但**特征提取器
> SuperPoint 的代码和权重都来自这个仓库**（`pip install -e` 指向它）。删掉会直接
> `ModuleNotFoundError: No module named 'lightglue'`。
> SuperPoint 权重首次运行自动下载（约 50 MB）到 `C:\Users\<你>\.cache\torch\hub\checkpoints`。

---

## 二、使用

```powershell
conda activate lightglue
cd E:\poster\poster_match

# 建参考图特征库（约 30 秒；参考图增删后重跑）
python build_db.py --ref-dir "E:\poster\VRChat电影馆" --out db\fast.pt --scales 1024 700 480 320 220 --max-kp 1024

# 识别
python match_fast.py --db db\fast.pt --query "E:\poster\Test_Set" --out-csv out\fast.csv --viz-dir out\viz_fast

# 单张 + 分阶段耗时
python match_fast.py --db db\fast.pt --query "E:\poster\Test_Set\438100_61.jpg" --bench

# 有标准答案时算准确率（加 --lang en 可切换成英文报告）
python score.py --results out\fast.csv --labels labels_example.csv
```

输出：终端每行 `文件名 -> 参考图(内点数)`（一屏多张海报会全部列出）；
`out/fast.csv` 每张截图的 Top-5（`inliers` / `confidence` / `detected`）；
`out/viz_fast/` 在原图上画出四边形 + 文件名。

---

## 三、流程

```
① 读图 + 全图 CLAHE            13 ms   ← 只为下一步找框服务
② Canny+轮廓找"像画框的四边形"  14 ms   ← 通常 4 个，再加"整张图"兜底
③ 每框：裁剪→局部CLAHE→放大到1024→抽特征   92 ms   ← 最花时间
④ 每框：对描述子大表投一次票     27 ms   ← 一次融合 GEMM
⑤ 按票数高低花全局预算做验证     14 ms   ← 平均只验 3~4 次
⑥ 内点数排序 + 同位置 IoU 去重   13 ms
                              ──────
                              176 ms
```

难点不在"像不像"，而在**尺度**：海报在 1920 px 截图里往往只有 80–250 px，
参考图有 1000+ px，差 3–8 倍，再叠加透视、阴影、夜间暗光。四个关键设计：

1. **裁剪之后再做 CLAHE**（不是全图 CLAHE）。一张被强光冲淡的小图在整帧里只占约 2% 面积，
   全局归一化时影响被平均掉、救不回来；裁下来之后**它就是整张图**，这时拉对比度才有效。
   实测 `_51`/`_59`/`_61` 从"完全找不到"变成 258–522 内点。
2. **参考库存 5 个尺度**。描述子对尺寸很敏感，库里必须"每档都备一份"。实测同一张截图：
   单尺度库下正确答案的投票排名是**第 28 名**（最高票仅 3 票，纯噪声），
   5 尺度库下是**第 0 名（34 票 vs 次名 4 票）**。
3. **检索合成一次融合 GEMM**。把 1635 条候选的描述子摞成一张大表做一次大矩阵乘法，
   代替 1635 次由 Python 驱动的小矩阵乘法：**706 ms → 20 ms**（同样的算术量）。
   投票用的表还可以很薄（每条候选只放最强的 256 个描述子），因为投票只需排名次。
4. **验证预算全局分配**。不给每个框固定配额，而是把所有框的候选混在一起按票数排队花预算。
   空墙框的候选票数都低、排在最后，预算轮不到它们；而小缩略图上"票数不高但正确"的候选
   仍能拿到机会。这一步让验证从 **174 ms → 14 ms**，且准确率还上升了。

另外：fp16 抽特征（默认开）把抽特征从 157 降到 89 ms，准确率无变化。

---

## 四、判定阈值

| | 数值 |
| --- | --- |
| 真匹配的内点数 | 12 – 592 |
| 无海报画面的最高内点数 | 6 |
| 默认 `--min-inliers` | 10 |
| 默认 `--high-inliers`（标 high 置信度） | 60 |

**判据必须用内点数，不能用内点率。** 误匹配的内点率也能到 0.6–0.9——RANSAC 总能在少量
匹配里找出一个自洽子集——但绝对内点数分得很干净。

库里存在**内容极相似的参考图**，它们会同时上榜（19–66 内点）但都排在正确答案之下，
并被第⑥步的同位置 IoU 抑制掉，不会变成错误结果：

* `train_arrival_01` / `train_arrival_02`：同一部片的两张相似剧照。
* `moon_trip_02` 那张海报里**嵌了** `moon_trip_01` 的月亮剧照。
* 完全重复的图：`era4/3_Rashomon_1950_IMDb.png` 与 `era4/rashomon_02.jpg`、
  `era7/3__2_Chiphell_.png` 与 `era7/WanderingEarth_04_digitalcard.png`（分数会完全相同）。

---

## 五、参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--budget` | 32 | 每张截图最多几次验证，按票数从高到低花。验证很便宜，调大几乎不涨耗时 |
| `--max-props` | 4 | 候选框数量。一屏超过 4 张画时调大 |
| `--crop-target` | 1024 | 每个 crop 放大到的长边。**别调小**，掉召回且几乎不省时间 |
| `--vote-kp` / `--bank-kp` | 512 / 256 | 投票用的查询关键点数 / 每条参考在投票表里的描述子数 |
| `--verify` | 4 | 每个框最多进入队列的候选数（上限，不是必然花掉） |
| `--early-exit` | 80 | 某框拿到这么多内点就认为已确定，不再为它花预算 |
| `--ratio` | 0.90 | Lowe 比率检验阈值，越小越严 |
| `--min-inliers` / `--high-inliers` | 10 / 60 | 判定阈值 / high 置信度阈值 |
| `--iou-thresh` | 0.30 | 同位置去重的重叠阈值 |
| `--det-width` | 0 | 找边缘的分辨率，0=原图。**别改**，见下 |
| `--fp32` | — | 关掉半精度抽特征 |
| `--bench` | — | 打印分阶段平均耗时。改参数前先跑这个 |
| `--device` | cuda | 无显卡填 `cpu`（会慢很多） |

---

## 六、每个环节用的什么算法（出处）

整条流水线里**只有一个神经网络**（SuperPoint），其余全是经典算法。
OpenCV 在这里只提供经典算法的实现，它没有贡献任何"深度模型"。

| 环节 | 用的什么 | 深度学习？ | 出处 |
| --- | --- | --- | --- |
| 抽特征点 + 描述子 | **SuperPoint** | **是** | DeTone / Malisiewicz / Rabinovich, CVPRW, [arXiv:1712.07629](https://arxiv.org/abs/1712.07629)（Magic Leap）。实现与权重均来自 **[cvg/LightGlue](https://github.com/cvg/LightGlue)** 仓库的 release |
| 找候选画框 | Canny → findContours → approxPolyDP | 否 | Canny 1986；Douglas–Peucker 1973。OpenCV |
| 局部提亮 | CLAHE | 否 | Zuiderveld, Graphics Gems IV, 1994。OpenCV |
| 检索投票 | 全局最近邻投票（一次 GEMM） | 否 | 本项目自己写的（`bank_votes`）。思路是 image retrieval 里的 local-feature voting，可追到 Sivic & Zisserman "Video Google" (ICCV 2003) 那一脉 |
| 配对 | Mutual-NN + Lowe ratio test | 否 | ratio test 出自 Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004（即 SIFT 论文） |
| 几何校验 | RANSAC（USAC_MAGSAC）+ findHomography | 否 | RANSAC: Fischler & Bolles 1981；MAGSAC++: Barath et al., CVPR 2020。OpenCV |

这个领域的标准流水线是三级：**① 找点+描述 → ② 配对 → ③ 几何校验**。
本项目第②级用的是经典方法，但第①级的深度描述子没动——所以它**不是"纯传统 CV"**，
最关键的表征仍然是神经网络产出的。

### 实测：把神经网络换成传统 SIFT

其他一切不变，只把第①级从 SuperPoint 换成 OpenCV SIFT：

| 第①级 | 完全正确 | 误报 | precision | recall | F1 | 单张耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| **SuperPoint**（默认） | **36/39** | **0** | **1.000** | **0.923** | **0.960** | **~170 ms** |
| SIFT（传统） | 35/39 | 1 | 0.972 | 0.897 | 0.933 | 389 ms |

结论说实在话：**SIFT 只差一点点**（少 1 张、多 1 个误报），但慢 2.2 倍（OpenCV SIFT 只能跑 CPU）。
也就是说准确率主要是**架构**给的（裁剪放大对齐尺度 + 多尺度库 + 几何校验），
SuperPoint 的贡献是"锦上添花 + 速度"。复现：

```powershell
python build_db.py --ref-dir "E:\poster\VRChat电影馆" --out db\sift.pt --extractor sift --scales 1024 700 480 320 220 --max-kp 1024
python match_fast.py --db db\sift.pt --query "E:\poster\Test_Set" --out-csv out\sift.csv
python score.py --results out\sift.csv --labels labels_example.csv
```

（`db\sift.pt` 有 307 MB，比对完删掉即可。`build_db.py` 也支持 `--extractor aliked / disk`。）

---

## 七、走过的弯路（别重复踩）

这个项目最初是 **13 秒/张**的版本（全图分块抽特征 + 300 次 LightGlue 匹配），
准确率 37/39。现在的实时版是 158–176 ms、36/39——**快 80 倍，少 1 张**。
那份慢代码已按需求删除，但结论值得留着：

**瓶颈不是模型，是干法。** 当初的耗时分布：

```
LightGlue + RANSAC × 300 对    6068 ms   ← 每对 20 ms
全图 4×4 分块抽特征            1012 ms
1635 条候选逐个粗筛             706 ms
```

就算把匹配器换成**无限快**，剩下的 1012 + 706 还有 1.7 秒，照样进不了 0.3 秒。
80 倍提速里**换模型贡献 0**，全部来自"少调几次 + 把碎活合成一次大活"。

**试过但更差、已回退的四件事**（都听起来很合理）：

| 想法 | 实测结果 |
| --- | --- |
| `torch.compile(mode="reduce-overhead")` 加速 LightGlue | **58 ms/次**（原本 19 ms）。要 padding 到静态长度且关掉剪枝。附带装了 `triton-windows`，留着无害 |
| 候选框按"内部纹理密度"排序（想优先小照片而不是大空白展板） | 36 → **33**。单位面积梯度最高的是**阴影拐角**，不是照片 |
| 边缘检测降分辨率（`--det-width 960`）省时间 | 3 张截图直接变成 **0 个候选框**（细边框糊掉了）。只省 10 ms，不值 |
| 缩小 `--crop-target` 到 800、砍关键点数到 768 | 掉 3 个命中，且几乎不省时间（验证成本不由关键点数决定） |

还有一条关键实测：单次 LightGlue **19 ms，而 RANSAC 只要 0.08 ms**——瓶颈完全在 LightGlue
自己（1024 个点这个规模是 kernel launch 受限的，所以砍关键点没用）。
把第②级换成 Mutual-NN 后单次约 0.5 ms，省下的预算拿去验证更多候选，
结果**又快又更准**（36 vs 用 LightGlue 验证的 34）。

---

## 八、想更快 / 更准：可以换的现成项目

当前 176 ms 里最大一块是**抽特征 92 ms**（SuperPoint，约 5 个 crop × 18 ms）。按性价比排序：

1. **[verlab/accelerated_features](https://github.com/verlab/accelerated_features)（XFeat, CVPR 2024）**
   —— 最值得换。专为实时设计，比 SuperPoint 快约 5 倍且能在 CPU 上跑；描述子只有 64 维
   （投票表直接小 4 倍）。**本机已装的 `kornia 0.8.3` 里就有 `kornia.feature.XFeat`**，不用另外装。
   换它需要在 `lgtools.build_extractor` 里加一路、并重建一次特征库，预计能进 60–80 ms。
2. **[zju3dv/EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR)**
   —— semi-dense、约 30 ms/对，对**低纹理和极小目标**比稀疏方法鲁棒得多。
   如果目标是把 `_51` 那类 75 px 缩略图也啃下来，这个比换检测器更对症
   （代价是每对 30 ms，得靠投票把候选压到 1–2 个）。
3. **[fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)**
   —— 如果哪天想把 LightGlue 请回来做验证，走 ONNX / TensorRT 是绕过 kernel launch 开销的正解
   （官方报 2–4 倍）。kornia 也有 `kornia.feature.OnnxLightGlue`。
4. **[cvg/Hierarchical-Localization (hloc)](https://github.com/cvg/Hierarchical-Localization)**
   —— "全局描述子检索 → ANN 候选 → 局部匹配"的参考架构，也就是本项目第 3 条设计的标准做法。
   参考库要从 327 张扩到上万张时照它改，那时再上
   **[DINOv2](https://github.com/facebookresearch/dinov2)** / MixVPR / SALAD + **faiss** 做检索前端。

---

## 九、文件

```
poster_match/
├─ 新手指南.md          通俗版：架构讲解 + 代码逐步详解 + 怎么换自己的测试集
├─ README.md            本文件：参数表、实测数据、算法出处、走过的弯路
├─ lgtools.py           公共库：读图(支持中文路径)、特征库、描述子大表+投票、
│                       候选框、裁剪放大、Mutual-NN+RANSAC 验证、画框
├─ build_db.py          建参考图特征库
├─ match_fast.py        识别主程序
├─ score.py             拿标准答案给结果打分（precision / recall / F1）
├─ labels_example.csv   Test_Set 的标准答案（人工核对过），也可当格式样例
├─ db/fast.pt           特征库（560 MB，1635 条 = 327 张 × 5 尺度）
└─ out/                 fast.csv + viz_fast/
```

> Windows 上 `cv2.imread` **打不开中文路径**（返回 None），`VRChat电影馆` 正好是中文目录，
> 所以 `lgtools.py` 里统一走 `cv2.imdecode(np.fromfile(...))`，改路径时别换回 `imread`。
