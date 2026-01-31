from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


VIDEO_PATH = "analysis_git200/data/comprec.mp4"
TIDY_PATH  = "analysis_git200/data/processed_dataset/trajectories_tidy.parquet"
OUT_DIR    = Path("analysis_git200/data/processed_dataset/visualizations/out_per_episode")

RAW_COLOR   = (0, 255, 255)   # желтый
FILT_COLOR  = (0, 0, 255)     # красный
START_COLOR = (0, 255, 0)     # зеленый
END_COLOR   = (255, 0, 0)     # синий

RAW_THICK  = 3
FILT_THICK = 3
POINT_R    = 3
DRAW_POINTS_EVERY = 3  


DRAW_EVERY_N = 1  


def extract_first_frame_ffmpeg(video_path: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", str(out_png)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(err)


def _finite_xy(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[ok]
    ys = ys[ok]
    return xs, ys

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

def draw_points(img_bgr: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: tuple, r: int, every: int = 3) -> int:
    xs, ys = _finite_xy(xs, ys)
    if len(xs) < 1:
        return 0
    if every > 1:
        xs = xs[::every]
        ys = ys[::every]
    for x, y in zip(xs, ys):
        cv2.circle(img_bgr, (int(round(x)), int(round(y))), r, color, -1, lineType=cv2.LINE_AA)
    return len(xs)

def draw_start_end(img_bgr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    xs, ys = _finite_xy(xs, ys)
    if len(xs) < 1:
        return
    p0 = (int(round(xs[0])), int(round(ys[0])))
    p1 = (int(round(xs[-1])), int(round(ys[-1])))
    cv2.circle(img_bgr, p0, POINT_R + 2, START_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.circle(img_bgr, p1, POINT_R + 2, END_COLOR, -1, lineType=cv2.LINE_AA)



def debug_xy(name: str, xs: np.ndarray, ys: np.ndarray, W: int, H: int) -> str:
    xs, ys = _finite_xy(xs, ys)
    if len(xs) == 0:
        return f"{name}: no finite points"
    mnx, mxx = float(xs.min()), float(xs.max())
    mny, mxy = float(ys.min()), float(ys.max())
    inside = (mnx <= W and mxx >= 0 and mny <= H and mxy >= 0)
    return f"{name}: n={len(xs)} x=[{mnx:.1f},{mxx:.1f}] y=[{mny:.1f},{mxy:.1f}] inside_frame={inside}"



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) фон
    bg_path = OUT_DIR / "background_000001.png"
    extract_first_frame_ffmpeg(VIDEO_PATH, bg_path)

    bg = cv2.imread(str(bg_path))
    if bg is None:
        raise RuntimeError(f"Не удалось прочитать фон: {bg_path}")
    H, W = bg.shape[:2]

    tidy = pd.read_parquet(TIDY_PATH)

    if "keep" in tidy.columns:
        tidy = tidy[tidy["keep"]].copy()

    cols = set(tidy.columns)
    has_raw = ("raw_x" in cols) and ("raw_y" in cols)
    has_filt = ("filt_x" in cols) and ("filt_y" in cols)
    has_smooth = ("x_s" in cols) and ("y_s" in cols)


    if has_smooth:
        fx_col, fy_col = "x_s", "y_s"
    elif has_filt:
        fx_col, fy_col = "filt_x", "filt_y"
    else:
        raise RuntimeError("В trajectories_tidy.parquet нет x_s/y_s и нет filt_x/y. Нечего рисовать.")

    print(f"[INFO] Columns: raw={has_raw}, filt={has_filt}, smooth={has_smooth}. Using filtered=({fx_col},{fy_col})")
    print(f"[INFO] Episodes: {tidy['episode'].nunique()}")

    for ep, g in tidy.groupby("episode", sort=True):
        canvas = bg.copy()

        if has_raw:
            raw_x = g["raw_x"].to_numpy(dtype=float)
            raw_y = g["raw_y"].to_numpy(dtype=float)
        else:
            raw_x = np.array([], dtype=float)
            raw_y = np.array([], dtype=float)

        # filtered (x_s/y_s или filt_x/y)
        fx = g[fx_col].to_numpy(dtype=float)
        fy = g[fy_col].to_numpy(dtype=float)

        if ep < (tidy["episode"].min() + 3):
            print(f"[DBG ep={ep}] {debug_xy('RAW', raw_x, raw_y, W, H)}")
            print(f"[DBG ep={ep}] {debug_xy('FILT', fx, fy, W, H)}")

        if has_raw:
            draw_polyline(canvas, raw_x, raw_y, RAW_COLOR, RAW_THICK)
            draw_points(canvas, raw_x, raw_y, RAW_COLOR, r=2, every=DRAW_POINTS_EVERY)

        draw_polyline(canvas, fx, fy, FILT_COLOR, FILT_THICK)
        draw_points(canvas, fx, fy, FILT_COLOR, r=2, every=DRAW_POINTS_EVERY)

        draw_start_end(canvas, fx, fy)

        label = f"episode {ep} | raw={'yes' if has_raw else 'no'} | filt={fx_col}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(canvas, (15, 10), (15 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
        cv2.putText(canvas, label, (20, 10 + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        out_path = OUT_DIR / f"ep_{ep:06d}.png"
        cv2.imwrite(str(out_path), canvas)

    print(f"[OK] Saved overlays to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()