"""EPSILON ANIMATION: render the six-panel figure (u_x | dCFS | von Mises;
map view + cross-section) over a ladder of mollification parameters eps and
assemble the frames into an MP4 with ffmpeg.

The story: the ELASTIC field is eps-stable.  Stress panels subtract the
anelastic eigenstress C:eps_star by default (inherited from
demo_sixpanel.compute_fields), so as eps shrinks the on-fault dCFS/von-Mises
converge instead of blowing up ~1/eps; the only eps-dependent features are
the genuine (eps-capped) tip concentrations sharpening at the patch edges
and the O(eps) near-trace convergence of the far field.  Pass --total for
the old raw-TOTAL movie (the fault zone narrowing/intensifying as the
eigenstress grows — deprecated fault-zone reading, comparison only).
COLOR LIMITS ARE FROZEN across the ladder so what changes between frames is
a real signal, not a rescaling artifact.  Limits come from a cheap PRE-PASS
(a log-spaced subset of the ladder at ~1/3 grid resolution — the limits are
percentile statistics and are insensitive to grid resolution), after which
frames STREAM into <out>_frames/ as soon as they are computed, in parallel
across worker processes — watch the directory fill up.

Run:
    python mhf/demo_eps_animation.py                       # 2.0 -> 0.25 km, 8 frames
    python mhf/demo_eps_animation.py --eps-values 2 1 0.5 0.25 --fps 1
    python -m mhf.demo_eps_animation --eps-max 0.001 --eps-min 2 \
        --n-frames 200 --fps 10 --n 363 --ny 483 --nz 243   # 1 m -> 2 km, hi-res

Output: mhf/<out>_frames/frame_###.png and mhf/<out>.mp4 (requires ffmpeg on
PATH; if absent, the frames are kept and the ffmpeg command is printed).
"""
import argparse
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("MPLBACKEND", "Agg")     # workers render headless

import numpy as np

try:                                   # python -m mhf.demo_eps_animation
    from .demo_sixpanel import (add_common_args, compute_fields, panel_limits,
                                render_sixpanel)
except ImportError:                    # python mhf/demo_eps_animation.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mhf.demo_sixpanel import (add_common_args, compute_fields,
                                   panel_limits, render_sixpanel)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Animate the six-panel figure over an eps ladder "
                     "(frozen color limits, streaming parallel frames) and "
                     "assemble an MP4 with ffmpeg."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p)
    p.add_argument("--eps-values", type=float, nargs="+", default=None,
                   help="explicit eps list (km), rendered in the given order; "
                        "overrides --eps-max/--eps-min/--n-frames")
    p.add_argument("--eps-max", type=float, default=2.0,
                   help="ladder start (km); may be smaller than --eps-min "
                        "for an ascending ladder")
    p.add_argument("--eps-min", type=float, default=0.25,
                   help="ladder end (km)")
    p.add_argument("--n-frames", type=int, default=8,
                   help="number of frames, log-spaced from eps-max to eps-min")
    p.add_argument("--fps", type=float, default=10.0,
                   help="animation frame rate (frames/s)")
    p.add_argument("--hold", type=int, default=2,
                   help="repeat the first and last frame this many extra "
                        "times (lets the endpoints sink in)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default: min(6, cpu-2); "
                        "each worker peaks at ~3.5 GB on 3x grids — scale "
                        "workers to available RAM)")
    p.add_argument("--prepass-frames", type=int, default=12,
                   help="frames in the coarse limits pre-pass")
    p.add_argument("--out", default="anim_mhf_sixpanel_eps",
                   help="output stem: <out>.mp4 + <out>_frames/, in mhf/")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Worker tasks (module-level so they pickle under spawn)
# ---------------------------------------------------------------------------
def _prepass_task(payload):
    eps, args_dict = payload
    ns = argparse.Namespace(**args_dict)
    F = compute_fields(ns, eps)
    return {row: F[row] for row in ("top", "bot")}     # only what limits need


def _frame_task(payload):
    k, eps, args_dict, limits, frames_dir = payload
    ns = argparse.Namespace(**args_dict)
    F = compute_fields(ns, eps)
    render_sixpanel(ns, F, eps, os.path.join(frames_dir, f"frame_{k:03d}"),
                    limits=limits, exts=("png",))
    return k, eps


def _coarse_dict(args_dict, factor=3):
    """Copy of the run config at ~1/factor grid resolution (odd counts keep
    the y=0 on-fault grid line, so the percentile statistics see the core)."""
    d = dict(args_dict)
    d["n"] = max(41, (d["n"] // factor) | 1)
    d["ny"] = max(41, (d["ny"] // factor) | 1)
    d["nz"] = max(21, (d["nz"] // factor) | 1)
    return d


def main(argv=None):
    args = _parse_args(argv)
    if args.eps_values:
        eps_list = list(args.eps_values)
    else:
        eps_list = list(np.geomspace(args.eps_max, args.eps_min, args.n_frames))
    workers = args.workers or min(6, max(1, (os.cpu_count() or 4) - 2))
    args_dict = vars(args)

    here = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(here, f"{args.out}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"eps ladder ({len(eps_list)} frames): {eps_list[0]:.3g} ... "
          f"{eps_list[-1]:.3g} km;  {workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # ---- pre-pass: frozen limits from a coarse log-spaced subset -------
        n_pre = min(args.prepass_frames, len(eps_list))
        idx = np.unique(np.linspace(0, len(eps_list) - 1, n_pre).astype(int))
        coarse = _coarse_dict(args_dict)
        print(f"limits pre-pass: {len(idx)} coarse frames ...", flush=True)
        pre = list(pool.map(_prepass_task,
                            [(eps_list[i], coarse) for i in idx]))
        limits = panel_limits(pre, log_pct=99.0)
        print(f"  frozen limits: u_x ±{limits[0]:.3g} mm, "
              f"dCFS ±{limits[1]:.3g} MPa, sigma_vM "
              f"[{limits[2][0]:.3g}, {limits[2][1]:.3g}] MPa (log)", flush=True)

        # ---- streaming frames: render as soon as each is computed ----------
        print(f"rendering {len(eps_list)} frames into {frames_dir} ...",
              flush=True)
        futures = [pool.submit(_frame_task,
                               (args.hold + i, e, args_dict, limits, frames_dir))
                   for i, e in enumerate(eps_list)]
        done = 0
        for fut in as_completed(futures):
            k, e = fut.result()
            done += 1
            print(f"  [{done}/{len(eps_list)}] frame_{k:03d}  (eps={e:.3g} km)",
                  flush=True)

    # ---- endpoint holds (file copies) --------------------------------------
    first = os.path.join(frames_dir, f"frame_{args.hold:03d}.png")
    last = os.path.join(frames_dir,
                        f"frame_{args.hold + len(eps_list) - 1:03d}.png")
    for h in range(args.hold):
        shutil.copy(first, os.path.join(frames_dir, f"frame_{h:03d}.png"))
        shutil.copy(last, os.path.join(
            frames_dir, f"frame_{args.hold + len(eps_list) + h:03d}.png"))

    mp4 = os.path.join(here, f"{args.out}.mp4")
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(frames_dir, "frame_%03d.png"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # h264 needs even dims
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4]
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; frames are kept. Assemble with:")
        print("  " + " ".join(cmd), flush=True)
        return
    print("assembling mp4 ...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  wrote {mp4}", flush=True)


if __name__ == "__main__":
    main()
