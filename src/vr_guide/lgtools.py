"""Shared helpers for real-time poster retrieval.

Pipeline (see match_fast.py):
    detect picture-frame quads -> crop + zoom each -> SuperPoint features
    -> vote against a flat descriptor bank -> mutual-NN + RANSAC verification

Only SuperPoint (from the cvg/LightGlue package) is a neural network; the
proposal, retrieval and verification stages are classical algorithms.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# IO  (cv2.imread cannot open non-ASCII paths on Windows -> imdecode wrapper)
# --------------------------------------------------------------------------- #
def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"Empty or unreadable file: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"cv2 could not decode: {path}")
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"cv2 could not encode: {path}")
    buf.tofile(str(path))


def load_image_torch(path: str | Path) -> torch.Tensor:
    """BGR uint8 file -> RGB float CHW tensor in [0, 1] (SuperPoint convention)."""
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0


def list_images(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def build_extractor(name: str, max_kp: int, device: torch.device):
    from lightglue import ALIKED, DISK, SIFT, SuperPoint

    name = name.lower()
    cls = {"superpoint": SuperPoint, "aliked": ALIKED, "disk": DISK, "sift": SIFT}[name]
    return cls(max_num_keypoints=max_kp).eval().to(device)


@torch.no_grad()
def extract(extractor, img: torch.Tensor, device: torch.device, resize) -> dict:
    """resize = target long-edge length, or None to keep native resolution."""
    return extractor.extract(img.to(device), resize=resize)


def enhance_bgr(img: np.ndarray, clahe: bool = True) -> np.ndarray:
    """Local contrast boost — the museum has very dark night-lit rooms."""
    if not clahe:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# Feature DB
# --------------------------------------------------------------------------- #
def pack_feats(feats: dict) -> dict:
    """Strip batch dim, store compactly (fp16) for the on-disk DB."""
    return {
        "keypoints": feats["keypoints"][0].cpu().half(),
        "descriptors": feats["descriptors"][0].cpu().half(),
        "image_size": feats["image_size"][0].cpu().float(),
    }


def save_db(db: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(db, str(path))


def load_db(path: str | Path, device: torch.device) -> dict:
    db = torch.load(str(path), map_location="cpu")
    for e in db["entries"]:
        e["keypoints"] = e["keypoints"].to(device)
        e["descriptors"] = e["descriptors"].to(device)
        e["image_size"] = e["image_size"].to(device)
    return db


# --------------------------------------------------------------------------- #
# Retrieval: one fused GEMM against a flat descriptor bank
# --------------------------------------------------------------------------- #
def build_bank(entries: list[dict],
               per_entry: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten every entry's descriptors into one matrix + a row->entry index.

    Comparing the query against each entry separately means one tiny GEMM per
    entry driven from Python — measured at ~700 ms for 1635 entries. Stacking
    them and doing a single large GEMM over the same data takes ~20 ms, which is
    what makes real-time retrieval possible.

    per_entry > 0 keeps only that many descriptors per entry (they are stored in
    descending keypoint score, so these are the strongest). Retrieval only needs
    enough votes to rank references, while verification still uses the full set —
    which is what lets the bank stay multi-scale without paying for it. Scale
    coverage matters far more here than keypoint count: a 1.5x scale gap between
    query crop and reference is enough to make the vote pure noise (measured:
    correct answer ranked 28th with a single-scale bank, 1st with five scales).
    """
    descs = [e["descriptors"][:per_entry] if per_entry else e["descriptors"]
             for e in entries]
    bank = torch.cat(descs).half().contiguous()
    seg = torch.cat([
        torch.full((d.shape[0],), i, dtype=torch.int64, device=bank.device)
        for i, d in enumerate(descs)
    ])
    return bank, seg


@torch.no_grad()
def bank_votes(qdesc: torch.Tensor, bank: torch.Tensor, seg: torch.Tensor,
               n_entries: int, sim_thresh: float = 0.80,
               chunk: int = 200_000) -> torch.Tensor:
    """Each query keypoint votes for the entry holding its global nearest neighbour.

    Descriptors are L2-normalised by the extractor, so a dot product is the
    cosine similarity. Chunked over the bank only to bound peak memory.
    Returns a vote count per entry.
    """
    q = qdesc.half()
    best_sim = torch.full((q.shape[0],), -2.0, dtype=torch.half, device=q.device)
    best_row = torch.zeros(q.shape[0], dtype=torch.int64, device=q.device)
    for s in range(0, bank.shape[0], chunk):
        sim = q @ bank[s:s + chunk].T
        v, i = sim.max(dim=1)
        upd = v > best_sim
        best_sim = torch.where(upd, v, best_sim)
        best_row = torch.where(upd, i + s, best_row)
    keep = best_sim > sim_thresh
    return torch.bincount(seg[best_row[keep]], minlength=n_entries).float()


# --------------------------------------------------------------------------- #
# Proposals: find the framed pictures, then crop and zoom them
# --------------------------------------------------------------------------- #
def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two quads' bounding boxes — no rasterisation, ~1000x cheaper."""
    ax0, ay0, ax1, ay1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    bx0, by0, bx1, by1 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def propose_quads(img_bgr: np.ndarray, min_area: int = 500, max_area_frac: float = 0.45,
                  max_props: int = 14, det_width: int = 0) -> list[np.ndarray]:
    """Cheap class-agnostic 'framed picture' detector.

    Museum posters are planar quads on flat walls, so contour + polygon
    approximation finds them reliably — including the ~80 px thumbnails that are
    far too small to detect features on at native frame resolution.

    det_width > 0 runs the edge detection on a downscaled copy and scales the
    quads back up. Tempting but measured as harmful: at det_width=960 three test
    screenshots yielded *zero* proposals because the thin frame borders blur away.
    Default 0 (full resolution) costs only ~16 ms.
    """
    full_h, full_w = img_bgr.shape[:2]
    up = 1.0
    if det_width and full_w > det_width:
        up = full_w / det_width
        img_bgr = cv2.resize(img_bgr, (det_width, int(round(full_h / up))),
                             interpolation=cv2.INTER_AREA)
        min_area = max(60, int(min_area / (up * up)))
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    quads: list[np.ndarray] = []
    for lo in (20, 45, 90):
        edges = cv2.Canny(gray, lo, lo * 3)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area_frac * h * w:
                continue
            ap = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            q = ap.reshape(4, 2).astype(np.float32)
            side = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
            if side.min() < 12 or side.max() / max(side.min(), 1e-6) > 8:
                continue
            quads.append(q)

    # greedy NMS, largest first (texture-per-area ranking was tried and is worse:
    # it promotes tiny high-contrast shadow corners over actual pictures)
    quads.sort(key=lambda q: -cv2.contourArea(q))
    kept: list[np.ndarray] = []
    for q in quads:
        if all(bbox_iou(q, k) < 0.4 for k in kept):
            kept.append(q)
        if len(kept) >= max_props:
            break
    if up != 1.0:
        kept = [np.clip(q * up, [0, 0], [full_w - 1, full_h - 1]).astype(np.float32)
                for q in kept]
    return kept


def crop_around(img_bgr: np.ndarray, quad: np.ndarray, margin: float = 0.30,
                target: int = 1024, max_upscale: float = 10.0, clahe: bool = True):
    """Axis-aligned crop around a quad, upscaled so it rivals reference detail.

    Two things matter here and both were measured:

    * CLAHE is applied *after* cropping. A poster washed out by a light gradient
      is unrecoverable when normalised over the whole frame (it occupies ~2% of
      the pixels), but recovers fully once it *is* the whole image.
    * Zooming to `target` is what turns "small query vs large reference" into a
      like-for-like comparison. Lowering it loses matches and saves almost
      nothing, since extraction cost is dominated elsewhere.

    Returns (crop, origin, scale); origin/scale map crop coords back to the frame.
    """
    h, w = img_bgr.shape[:2]
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0 = int(max(0, np.floor(x0 - mx)))
    y0 = int(max(0, np.floor(y0 - my)))
    x1 = int(min(w, np.ceil(x1 + mx)))
    y1 = int(min(h, np.ceil(y1 + my)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None, None, None
    crop = img_bgr[y0:y1, x0:x1]
    if clahe:
        crop = enhance_bgr(crop, clahe=True)
    s = min(target / max(crop.shape[:2]), max_upscale)
    s = max(s, 1.0)
    if s > 1.0:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop, (x0, y0), s


# --------------------------------------------------------------------------- #
# Verification: mutual-NN + Lowe ratio + RANSAC homography
# --------------------------------------------------------------------------- #
def _quad_is_sane(quad: np.ndarray, qw: int, qh: int) -> bool:
    """Reject degenerate/imploded homographies (folded, tiny or huge quads)."""
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 100 or area > 4.0 * qw * qh:
        return False
    # convexity: all cross products of consecutive edges share a sign
    v = np.roll(quad, -1, axis=0) - quad
    cross = v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1)
    return bool(np.all(cross > 0) or np.all(cross < 0))


@torch.no_grad()
def match_one_mnn(qfeats: dict, entry: dict, ransac_thresh: float = 6.0,
                  ratio: float = 0.90, min_matches: int = 6,
                  ransac_iters: int = 2000) -> dict:
    """Verify one candidate: mutual nearest neighbours + Lowe ratio + RANSAC.

    LightGlue costs ~19 ms per pair on this GPU and is launch-bound, so it cannot
    be made much cheaper (torch.compile made it *worse*, at 58 ms). But by this
    point the crop is already scale-normalised and the shortlist is short — the
    classic descriptor-matching regime, where mutual nearest neighbours plus a
    ratio test suffice. One GEMM plus MAGSAC is ~0.5 ms, roughly 40x cheaper,
    and the wider shortlist it buys made the whole pipeline *more* accurate
    (36/39 vs 34/39 with LightGlue verification).

    The inlier count is the confidence score: RANSAC keeps only correspondences
    that a single perspective transform can explain, so coincidental descriptor
    matches do not survive.
    """
    q = qfeats["descriptors"][0].half()
    r = entry["descriptors"].half()
    res = {"n_matches": 0, "n_inliers": 0, "H": None, "quad": None}
    if q.shape[0] < 2 or r.shape[0] < 2:
        return res

    sim = q @ r.T
    v2, i2 = sim.topk(2, dim=1)
    # cosine -> squared euclidean for normalised descriptors: d2 = 2 - 2s
    d1 = (2.0 - 2.0 * v2[:, 0].float()).clamp_(min=0)
    d2 = (2.0 - 2.0 * v2[:, 1].float()).clamp_(min=1e-9)
    ok = d1 <= (ratio ** 2) * d2
    nn0 = i2[:, 0]
    nn1 = sim.argmax(dim=0)
    ok &= nn1[nn0] == torch.arange(q.shape[0], device=q.device)

    idx = ok.nonzero(as_tuple=True)[0]
    res["n_matches"] = int(idx.numel())
    if idx.numel() < min_matches:
        return res

    kq = qfeats["keypoints"][0][idx].cpu().numpy().astype(np.float32)
    kr = entry["keypoints"][nn0[idx]].float().cpu().numpy().astype(np.float32)

    # map reference -> query, so the quad lands in screenshot coordinates
    H, mask = cv2.findHomography(kr, kq, cv2.USAC_MAGSAC, ransac_thresh,
                                 maxIters=ransac_iters, confidence=0.9999)
    if H is None or mask is None:
        return res

    rw, rh = entry["image_size"].tolist()
    corners = np.array([[0, 0], [rw, 0], [rw, rh], [0, rh]], np.float32).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    qw, qh = qfeats["image_size"][0].tolist()
    if not _quad_is_sane(quad, int(qw), int(qh)):
        return res

    res["n_inliers"] = int(mask.sum())
    res["H"] = H
    res["quad"] = quad
    return res


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
_PALETTE = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0),
            (0, 0, 255), (255, 128, 0)]


def draw_detections(query_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    vis = query_bgr.copy()
    scale = max(vis.shape[1] / 1600.0, 0.6)
    for i, d in enumerate(dets):
        color = _PALETTE[i % len(_PALETTE)]
        quad = np.round(d["quad"]).astype(np.int32)
        cv2.polylines(vis, [quad], True, color, max(2, int(3 * scale)), cv2.LINE_AA)
        label = f"{d['name']}  ({d['n_inliers']})"
        fs, th = 0.55 * scale, max(1, int(1.4 * scale))
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x = int(np.clip(quad[:, 0].min(), 0, vis.shape[1] - tw - 6))
        y = int(np.clip(quad[:, 1].min() - 6, tht + base + 4, vis.shape[0] - 4))
        cv2.rectangle(vis, (x - 3, y - tht - base - 2), (x + tw + 3, y + 2), (0, 0, 0), -1)
        cv2.putText(vis, label, (x, y - base + 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    color, th, cv2.LINE_AA)
    return vis
