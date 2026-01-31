from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

VIDEO_PATH = "analysis_git200/data/comprec.mp4"
TIDY_PATH  = "analysis_git200/data/processed_dataset/trajectories_tidy.parquet"
OUT_DIR    = Path("analysis_git200/data/processed_dataset/visualizations")

USE_ALPHA = True
ALPHA_RAW  = 0.55
ALPHA_FILT = 0.85


RAW_COLOR   = (0, 255, 255)   # желтый
FILT_COLOR  = (0, 0, 255)     # красный


RAW_THICK  = 1
FILT_THICK = 1


DRAW_EVERY_N = 1 

DRAW_POINTS = False
POINT_EVERY = 4
POINT_R = 1


def extract_first_frame_ffmpeg(video_path: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", str(out_png)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(err)


def _finite_xy(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(xs) & np.isfinite(ys)
    return xs[ok], ys[ok]


def draw_polyline(img_bgr: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: tuple, thickness: int) -> int:
    xs, ys = _finite_xy(xs, ys)
    if len(xs) < 2:
        return 0

    if DRAW_EVERY_N > 1:
        xs = xs[::DRAW_EVERY_N]
        ys = ys[::DRAW_EVERY_N]
        if len(xs) < 2:
            return 0

    pts = np.stack([xs, ys], axis=1).round().astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_bgr, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return len(xs)


def draw_points(img_bgr: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: tuple):
    xs, ys = _finite_xy(xs, ys)
    if len(xs) < 1:
        return
    xs = xs[::POINT_EVERY]
    ys = ys[::POINT_EVERY]
    for x, y in zip(xs, ys):
        cv2.circle(img_bgr, (int(round(x)), int(round(y))), POINT_R, color, -1, lineType=cv2.LINE_AA)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # фон
    bg_path = OUT_DIR / "background_000001.png"
    extract_first_frame_ffmpeg(VIDEO_PATH, bg_path)

    bg = cv2.imread(str(bg_path))
    if bg is None:
        raise RuntimeError(f"Не удалось прочитать фон: {bg_path}")

    # траектории
    tidy = pd.read_parquet(TIDY_PATH)
    if "keep" in tidy.columns:
        tidy = tidy[tidy["keep"]].copy()

    cols = set(tidy.columns)
    has_raw = ("raw_x" in cols) and ("raw_y" in cols)
    has_smooth = ("x_s" in cols) and ("y_s" in cols)
    has_filt = ("filt_x" in cols) and ("filt_y" in cols)

    # приоритетно x_s/y_s, иначе filt_x/y
    if has_smooth:
        fx_col, fy_col = "x_s", "y_s"
    elif has_filt:
        fx_col, fy_col = "filt_x", "filt_y"
    else:
        raise RuntimeError("В parquet нет x_s/y_s и нет filt_x/y — нечего рисовать.")

    print(f"[INFO] episodes: {tidy['episode'].nunique()} | raw={has_raw} | filtered=({fx_col},{fy_col})")

    if USE_ALPHA:
        overlay_raw = bg.copy()
        overlay_filt = bg.copy()
    else:
        overlay_raw = bg
        overlay_filt = bg

    for ep, g in tidy.groupby("episode", sort=False):
        # raw
        if has_raw:
            rx = g["raw_x"].to_numpy(dtype=float)
            ry = g["raw_y"].to_numpy(dtype=float)
            draw_polyline(overlay_raw, rx, ry, RAW_COLOR, RAW_THICK)
            if DRAW_POINTS:
                draw_points(overlay_raw, rx, ry, RAW_COLOR)

        # filtered
        fx = g[fx_col].to_numpy(dtype=float)
        fy = g[fy_col].to_numpy(dtype=float)
        draw_polyline(overlay_filt, fx, fy, FILT_COLOR, FILT_THICK)
        if DRAW_POINTS:
            draw_points(overlay_filt, fx, fy, FILT_COLOR)

    if USE_ALPHA:
        out = cv2.addWeighted(overlay_raw, ALPHA_RAW, bg, 1.0 - ALPHA_RAW, 0)
        out = cv2.addWeighted(overlay_filt, ALPHA_FILT, out, 1.0 - ALPHA_FILT, 0)
    else:
        out = bg

    legend = f"bundle: RAW(yellow) + FILTERED(red:{fx_col}) | eps={tidy['episode'].nunique()}"
    (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (15, 10), (15 + tw + 12, 10 + th + 14), (0, 0, 0), -1)
    cv2.putText(out, legend, (20, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out_path = OUT_DIR / "trajectory_bundle_raw_plus_filtered.png"
    cv2.imwrite(str(out_path), out)
    print(f"[OK] Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()