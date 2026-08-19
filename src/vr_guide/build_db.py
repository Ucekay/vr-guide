"""Extract local features for every reference poster and cache them to a .pt DB.

Example:
    python build_db.py --ref-dir "E:/poster/VRChat电影馆" --out db/superpoint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from vr_guide import lgtools as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, help="folder of reference posters (recursive)")
    ap.add_argument("--out", required=True, help="output .pt file")
    ap.add_argument("--extractor", default="superpoint",
                    choices=["superpoint", "aliked", "disk", "sift"])
    ap.add_argument("--max-kp", type=int, default=2048)
    ap.add_argument("--scales", type=int, nargs="+", default=[1024, 640],
                    help="long-edge sizes; several = scale pyramid, more robust to "
                         "posters appearing small in the screenshot")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ref_dir = Path(args.ref_dir)
    paths = T.list_images(ref_dir)
    if not paths:
        raise SystemExit(f"No images found under {ref_dir}")
    print(f"[db] {len(paths)} reference images x {len(args.scales)} scales "
          f"-> {len(paths) * len(args.scales)} entries")

    extractor = T.build_extractor(args.extractor, args.max_kp, device)

    entries: list[dict] = []
    for p in tqdm(paths, desc="extract", ncols=88):
        img = T.load_image_torch(p)
        rel = p.relative_to(ref_dir).as_posix()
        for s in args.scales:
            feats = T.extract(extractor, img, device, resize=s)
            e = T.pack_feats(feats)
            e["name"] = rel
            e["path"] = str(p)
            e["scale"] = s
            entries.append(e)

    db = {
        "entries": entries,
        "extractor": args.extractor,
        "max_kp": args.max_kp,
        "scales": args.scales,
        "ref_dir": str(ref_dir),
    }
    T.save_db(db, args.out)
    n_kp = sum(int(e["keypoints"].shape[0]) for e in entries)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"[db] saved {args.out}  ({len(entries)} entries, {n_kp} keypoints, {size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
