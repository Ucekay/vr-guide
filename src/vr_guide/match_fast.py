"""Identify which reference poster(s) a VRChat screenshot shows. ~158 ms/frame.

    detect picture-frame quads -> crop + zoom each -> extract features
    -> vote against the descriptor bank -> verify the best candidates

Design notes worth knowing before tuning (all measured, see README):

* The scale gap is the whole problem. A poster is 80-250 px inside a 1920 px
  frame while the reference is 1000+ px. Cropping and zooming each proposal is
  what makes the comparison like-for-like.
* Retrieval is one fused GEMM over a flat, multi-scale descriptor bank (~20 ms).
  Comparing entry by entry instead costs ~700 ms for the same arithmetic.
* Verification budget is global, not per crop, so blank-wall proposals cost
  almost nothing while a faint-but-correct vote still gets its chance.

Example:
    python match_fast.py --db db/fast.pt --query "E:/poster/Test_Set" --bench
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from vr_guide import lgtools as T


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--max-props", type=int, default=4, help="picture frames per screenshot")
    ap.add_argument("--det-width", type=int, default=0,
                    help="edge-detection resolution; 0 = full res (downscaling loses quads)")
    ap.add_argument("--crop-kp", type=int, default=1024, help="keypoints per crop")
    ap.add_argument("--verify", type=int, default=4,
                    help="entries verified per crop (upper bound; early exit usually stops at 1)")
    ap.add_argument("--early-exit", type=int, default=80,
                    help="stop verifying a crop once a candidate scores this many inliers")
    ap.add_argument("--ransac-iters", type=int, default=2000)
    ap.add_argument("--crop-target", type=int, default=1024,
                    help="long edge each crop is zoomed to before extraction")
    ap.add_argument("--vote-kp", type=int, default=512,
                    help="strongest N query keypoints used for retrieval voting")
    ap.add_argument("--ratio", type=float, default=0.90,
                    help="Lowe ratio test threshold; lower = stricter matching")
    ap.add_argument("--budget", type=int, default=32,
                    help="max matcher calls per screenshot, spent highest-vote first")
    ap.add_argument("--bank-kp", type=int, default=256,
                    help="descriptors per reference entry in the vote bank (0 = all)")
    ap.add_argument("--min-inliers", type=int, default=10,
                    help="mnn yields far fewer inliers than LightGlue; on this "
                         "data true matches score 12-590 and non-posters stay <=6")
    ap.add_argument("--high-inliers", type=int, default=60,
                    help="inliers for a 'high' confidence label")
    ap.add_argument("--iou-thresh", type=float, default=0.30)
    ap.add_argument("--ransac-thresh", type=float, default=6.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--viz-dir", default="")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="half-precision feature extraction")
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--bench", action="store_true", help="per-stage timing breakdown")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def bgr_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Upload uint8 and convert on the GPU.

    Doing cvtColor/transpose/float()/255 on the CPU first costs several ms per
    crop and ships 4x more bytes over PCIe; channel flip + scale is free on GPU.
    """
    t = torch.from_numpy(np.ascontiguousarray(img)).to(device, non_blocking=True)
    return t.permute(2, 0, 1).flip(0).float().div_(255.0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cuda = device.type == "cuda"

    db = T.load_db(args.db, device)
    entries, names = db["entries"], [e["name"] for e in db["entries"]]
    bank, seg = T.build_bank(entries, per_entry=args.bank_kp)
    print(f"[db] {len(entries)} entries / {bank.shape[0]} descriptors "
          f"({bank.numel() * 2 / 1e6:.0f} MB), scales={db['scales']}")

    extractor = T.build_extractor(db["extractor"], args.crop_kp, device)

    qpath = Path(args.query)
    queries = T.list_images(qpath) if qpath.is_dir() else [qpath]

    # warm up cuda kernels / autotune so the first timing is representative
    if queries:
        w = T.imread_unicode(queries[0])
        wf = T.extract(extractor, bgr_to_tensor(w[:512, :512], device), device, resize=512)
        T.bank_votes(wf["descriptors"][0], bank, seg, len(entries))
        T.match_one_mnn(wf, entries[0])
        if cuda:
            torch.cuda.synchronize()

    rows: list[dict] = []
    acc = {"read": 0.0, "propose": 0.0, "prep": 0.0, "extract": 0.0, "retrieve": 0.0, "verify": 0.0}
    total = 0.0

    def clk() -> float:
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    for qi, qp in enumerate(queries, 1):
        t_start = clk()

        t0 = clk()
        img_bgr = T.imread_unicode(qp)
        src = T.enhance_bgr(img_bgr, clahe=True)
        H, W = img_bgr.shape[:2]
        t1 = clk(); acc["read"] += t1 - t0

        props = T.propose_quads(src, max_props=args.max_props, det_width=args.det_width)
        # a poster filling much of the frame has no enclosing contour to find, and
        # dark rooms can defeat the edge detector entirely -> always try the frame
        props.append(np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32))
        t2 = clk(); acc["propose"] += t2 - t1

        pool: dict[str, dict] = {}
        crops: list[tuple] = []
        cands: list[tuple] = []          # (votes, crop index, entry index)
        for pq in props:
            crop, origin, s = T.crop_around(img_bgr, pq, margin=0.20,
                                            target=args.crop_target)
            if crop is None:
                continue
            ta = clk()
            ct = bgr_to_tensor(crop, device)
            tp = clk(); acc["prep"] += tp - ta
            if args.fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    cf = T.extract(extractor, ct, device, resize=None)
                # LightGlue is fed alongside fp32 reference features
                cf = {k: (v.float() if torch.is_floating_point(v) else v)
                      for k, v in cf.items()}
            else:
                cf = T.extract(extractor, ct, device, resize=None)
            tb = clk(); acc["extract"] += tb - tp

            # SuperPoint returns keypoints in descending score order, so the
            # leading rows are the strongest — plenty for a retrieval vote.
            votes = T.bank_votes(cf["descriptors"][0][:args.vote_kp], bank, seg,
                                 len(entries))
            # A vote ranks (reference, scale) jointly, so the shortlist holds
            # entries directly; no need to expand every scale of a named image.
            vv, vi = torch.topk(votes, min(args.verify, len(entries)))
            crops.append((cf, origin, s))
            cands += [(v, len(crops) - 1, int(i))
                      for v, i in zip(vv.tolist(), vi.tolist()) if v > 0]
            acc["retrieve"] += clk() - tb

        # Verification is the expensive stage, so spend a *global* budget on the
        # strongest evidence anywhere in the frame rather than a fixed quota per
        # crop: blank-wall proposals then cost nothing, and a faint-but-correct
        # vote on a tiny thumbnail still gets its matcher call.
        tc = clk()
        cands.sort(key=lambda c: -c[0])
        budget, done = args.budget, set()
        for v, ci, ei in cands:
            if budget <= 0:
                break
            if ci in done:
                continue
            cf, origin, s = crops[ci]
            r = T.match_one_mnn(cf, entries[ei],
                                ransac_thresh=args.ransac_thresh,
                                ratio=args.ratio,
                                ransac_iters=args.ransac_iters)
            budget -= 1
            if r["quad"] is None:
                continue
            nm = names[ei]
            if r["n_inliers"] >= args.early_exit:
                done.add(ci)          # this crop is settled; stop spending on it
            r = {**r, "name": nm, "scale": entries[ei]["scale"],
                 "quad": r["quad"] / s + np.array(origin, np.float32)}
            old = pool.get(nm)
            if old is None or r["n_inliers"] > old["n_inliers"]:
                pool[nm] = r
        acc["verify"] += clk() - tc

        ranked = sorted(pool.values(), key=lambda d: -d["n_inliers"])
        dets: list[dict] = []
        for r in ranked:
            if r["n_inliers"] < args.min_inliers:
                continue
            if any(T.bbox_iou(r["quad"], d["quad"]) > args.iou_thresh for d in dets):
                continue
            dets.append(r)

        dt = clk() - t_start
        total += dt
        head = ", ".join(f"{d['name']}({d['n_inliers']})" for d in dets) or "-- none --"
        print(f"[{qi:>3}/{len(queries)}] {qp.name:<18} {dt*1000:6.0f}ms "
              f"props={len(props):<2} -> {head}")

        for rank, r in enumerate(ranked[:args.topk], 1):
            n = r["n_inliers"]
            rows.append({
                "query": qp.name, "rank": rank, "ref": r["name"], "inliers": n,
                "matches": r["n_matches"],
                "confidence": "high" if n >= args.high_inliers
                              else "medium" if n >= args.min_inliers
                              else "low",
                "ref_scale": r["scale"],
                "detected": int(any(d["name"] == r["name"] for d in dets)),
            })

        if args.viz_dir and dets:
            T.imwrite_unicode(Path(args.viz_dir) / f"{qp.stem}.jpg",
                              T.draw_detections(img_bgr, dets))

    if args.out_csv and rows:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[out] {out}")

    n = len(queries)
    n_hit = len({r["query"] for r in rows if r["detected"]})
    print(f"[done] {n_hit}/{n} matched | mean {total / n * 1000:.0f} ms/query "
          f"({n / total:.1f} fps)")
    if args.bench:
        print("[bench] per-query average")
        for k, v in acc.items():
            print(f"    {k:<10}{v / n * 1000:7.1f} ms")
        other = total - sum(acc.values())
        print(f"    {'other':<10}{other / n * 1000:7.1f} ms")


if __name__ == "__main__":
    main()
