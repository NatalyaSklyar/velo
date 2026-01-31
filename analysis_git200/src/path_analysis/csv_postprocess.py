from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CFG:
    min_frames: int = 100
    min_duration_s: float = 1.0
    max_missing_xy_ratio: float = 0.30
    min_move_px: float = 1000.0
    min_path_len_px: float = 2000.0
    min_median_conf: float = 0.25
    min_detection_ratio: float = 0.15

    prefer_filtered: bool = True
    interp_limit: int = 8
    smooth_window: int = 5

    speed_clip_px_s: float = 10000.0

    max_step_px: float = 120.0
    max_speed_px_s: float = 2500.0
    max_jump_ratio: float = 0.02 
    max_tortuosity: float = 3.0


CSV_COLS = [
    "episode", "frame", "time",
    "raw_x", "raw_y", "filt_x", "filt_y",
    "conf",
    "roi_x1", "roi_y1", "roi_x2", "roi_y2",
    "cx", "cy", "rw", "rh"
]



def read_tracking_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=0,
        na_values=["", " ", "nan", "NaN", "None"],
        keep_default_na=True,
    )

    df = df[df["episode"].astype(str).str.lower() != "episode"].copy()

    # численные колонки
    float_cols = ["time", "conf", "raw_x", "raw_y", "filt_x", "filt_y"]
    int_cols = ["episode", "frame", "roi_x1", "roi_y1", "roi_x2", "roi_y2", "cx", "cy", "rw", "rh"]

    for c in float_cols + int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["episode", "frame"]).copy()
    df["episode"] = df["episode"].astype("int64")
    df["frame"] = df["frame"].astype("int64")

    for c in [c for c in int_cols if c not in ("episode", "frame")]:
        df[c] = df[c].astype("Int64")

    df = df.sort_values(["episode", "frame"]).reset_index(drop=True)
    return df


def choose_xy(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """
    - если prefer_filtered: filt -> raw -> NaN
    - иначе raw -> filt -> NaN
    """
    if cfg.prefer_filtered:
        x = df["filt_x"].copy()
        y = df["filt_y"].copy()
        x = x.where(~x.isna(), df["raw_x"])
        y = y.where(~y.isna(), df["raw_y"])
    else:
        x = df["raw_x"].copy()
        y = df["raw_y"].copy()
        x = x.where(~x.isna(), df["filt_x"])
        y = y.where(~y.isna(), df["filt_y"])

    out = df.copy()
    out["x"] = x
    out["y"] = y
    out["has_det"] = ~out["conf"].isna()
    out["has_xy"] = ~(out["x"].isna() | out["y"].isna())
    return out



def interp_and_smooth_episode(g: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    g = g.sort_values("frame").copy()

    if g["time"].is_monotonic_increasing and g["time"].nunique() > 1:
        idx = g["time"]
    else:
        idx = pd.RangeIndex(len(g))

    gx = g.set_index(idx)

    gx["x_i"] = gx["x"].interpolate(limit=cfg.interp_limit, limit_direction="both")
    gx["y_i"] = gx["y"].interpolate(limit=cfg.interp_limit, limit_direction="both")

    w = int(cfg.smooth_window)
    if w >= 3:
        gx["x_s"] = gx["x_i"].rolling(w, center=True, min_periods=1).median()
        gx["y_s"] = gx["y_i"].rolling(w, center=True, min_periods=1).median()
    else:
        gx["x_s"] = gx["x_i"]
        gx["y_s"] = gx["y_i"]

    gx = gx.reset_index(drop=True)
    gx["xy_filled"] = ~(gx["x_s"].isna() | gx["y_s"].isna())
    return gx



def episode_metrics(g: pd.DataFrame, cfg: CFG) -> dict:
    n = len(g)
    t0, t1 = float(g["time"].iloc[0]), float(g["time"].iloc[-1])
    dur = max(0.0, t1 - t0)

    det_ratio = float(g["has_det"].mean()) if n else 0.0
    med_conf = float(g["conf"].median()) if g["conf"].notna().any() else np.nan
    miss_ratio = float(1.0 - g["xy_filled"].mean()) if n else 1.0

    xs = g["x_s"].to_numpy(dtype=float)
    ys = g["y_s"].to_numpy(dtype=float)
    ts = g["time"].to_numpy(dtype=float)

    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)
    if valid.sum() >= 3:
        xv, yv, tv = xs[valid], ys[valid], ts[valid]

        dx = np.diff(xv)
        dy = np.diff(yv)
        dt = np.diff(tv)
        dt = np.where(dt <= 1e-9, np.nan, dt)

        step = np.sqrt(dx * dx + dy * dy)
        path_len = float(np.nansum(step))
        disp = float(np.sqrt((xv[-1] - xv[0]) ** 2 + (yv[-1] - yv[0]) ** 2))

        speed = step / dt
        speed = np.clip(speed, 0, cfg.speed_clip_px_s)

        med_speed = float(np.nanmedian(speed)) if np.isfinite(speed).any() else np.nan
        p95_speed = float(np.nanpercentile(speed[np.isfinite(speed)], 95)) if np.isfinite(speed).any() else np.nan

        max_step = float(np.nanmax(step)) if np.isfinite(step).any() else np.nan
        max_speed = float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan
        p99_speed = float(np.nanpercentile(speed[np.isfinite(speed)], 99)) if np.isfinite(speed).any() else np.nan

        bad_step = step > cfg.max_step_px
        bad_speed = speed > cfg.max_speed_px_s
        bad_mask = bad_step | bad_speed
        jump_ratio = float(np.nanmean(bad_mask)) if len(step) else 0.0

        tortuosity = float(path_len / max(disp, 1e-6))
    else:
        path_len = 0.0
        disp = 0.0
        med_speed = np.nan
        p95_speed = np.nan
        max_step = np.nan
        max_speed = np.nan
        p99_speed = np.nan
        jump_ratio = 1.0
        tortuosity = np.inf

    return {
        "episode": int(g["episode"].iloc[0]),
        "frames": int(n),
        "t_start": t0,
        "t_end": t1,
        "duration_s": dur,
        "det_ratio": det_ratio,
        "median_conf": med_conf,
        "missing_xy_ratio": miss_ratio,
        "displacement_px": disp,
        "path_len_px": path_len,
        "median_speed_px_s": med_speed,
        "p95_speed_px_s": p95_speed,
        "max_step_px": max_step,
        "p99_speed_px_s": p99_speed,
        "max_speed_px_s": max_speed,
        "jump_ratio": jump_ratio,
        "tortuosity": tortuosity,
    }


def is_good_episode(m: dict, cfg: CFG) -> bool:
    if m["frames"] < cfg.min_frames:
        return False
    if m["duration_s"] < cfg.min_duration_s:
        return False
    if np.isnan(m["median_conf"]) or m["median_conf"] < cfg.min_median_conf:
        return False
    if m["det_ratio"] < cfg.min_detection_ratio:
        return False
    if m["missing_xy_ratio"] > cfg.max_missing_xy_ratio:
        return False
    if m["displacement_px"] < cfg.min_move_px:
        return False
    if m["path_len_px"] < cfg.min_path_len_px:
        return False
    if np.isfinite(m["max_step_px"]) and m["max_step_px"] > cfg.max_step_px:
        return False
    if np.isfinite(m["max_speed_px_s"]) and m["max_speed_px_s"] > cfg.max_speed_px_s:
        return False
    if m["jump_ratio"] > cfg.max_jump_ratio:
        return False
    if np.isfinite(m["tortuosity"]) and m["tortuosity"] > cfg.max_tortuosity:
        return False

    return True


def reject_reason(m: dict, cfg: CFG) -> str:
    if m["frames"] < cfg.min_frames:
        return "too_few_frames"
    if m["duration_s"] < cfg.min_duration_s:
        return "too_short"
    if np.isnan(m["median_conf"]) or m["median_conf"] < cfg.min_median_conf:
        return "low_conf"
    if m["det_ratio"] < cfg.min_detection_ratio:
        return "few_detections"
    if m["missing_xy_ratio"] > cfg.max_missing_xy_ratio:
        return "too_many_gaps"
    if m["displacement_px"] < cfg.min_move_px:
        return "too_small_displacement"
    if m["path_len_px"] < cfg.min_path_len_px:
        return "too_short_path"
    if np.isfinite(m["max_step_px"]) and m["max_step_px"] > cfg.max_step_px:
        return "jump_max_step"
    if np.isfinite(m["max_speed_px_s"]) and m["max_speed_px_s"] > cfg.max_speed_px_s:
        return "jump_max_speed"
    if m["jump_ratio"] > cfg.max_jump_ratio:
        return "jump_ratio"
    if np.isfinite(m["tortuosity"]) and m["tortuosity"] > cfg.max_tortuosity:
        return "too_tortuous"

    return ""

def build_tidy_dataset(
    csv_path: str | Path,
    out_dir: str | Path,
    cfg: CFG = CFG(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_tracking_csv(csv_path)
    df = choose_xy(df, cfg)

    processed = []
    metrics = []

    for ep, g in df.groupby("episode", sort=True):
        gg = interp_and_smooth_episode(g, cfg)
        m = episode_metrics(gg, cfg)
        keep = is_good_episode(m, cfg)
        m["keep"] = keep
        m["reject_reason"] = "" if keep else reject_reason(m, cfg)

        gg["keep"] = keep
        processed.append(gg)
        metrics.append(m)

    tidy = pd.concat(processed, ignore_index=True)
    summary = pd.DataFrame(metrics).sort_values("episode").reset_index(drop=True)

    tidy_keep = tidy[tidy["keep"]].copy()

    dt = tidy_keep.groupby("episode")["time"].diff()
    tidy_keep["vx_px_s"] = tidy_keep.groupby("episode")["x_s"].diff() / dt
    tidy_keep["vy_px_s"] = tidy_keep.groupby("episode")["y_s"].diff() / dt
    tidy_keep["speed_px_s"] = np.sqrt(tidy_keep["vx_px_s"] ** 2 + tidy_keep["vy_px_s"] ** 2)
    tidy_keep["speed_px_s"] = tidy_keep["speed_px_s"].clip(lower=0, upper=cfg.speed_clip_px_s)

    tidy_path = out_dir / "trajectories_tidy.parquet"
    summary_path = out_dir / "episodes_summary.csv"
    rejected_path = out_dir / "episodes_rejected.csv"

    tidy_keep.to_parquet(tidy_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary[~summary["keep"]].to_csv(rejected_path, index=False)

    print(f"[OK] tidy trajectories: {tidy_path}  (rows={len(tidy_keep)})")
    print(f"[OK] summary:          {summary_path} (episodes={len(summary)})")
    print(f"[OK] rejected list:    {rejected_path} (episodes={(~summary['keep']).sum()})")

    return tidy_keep, summary



if __name__ == "__main__":
    cfg = CFG(
        min_frames=20,
        min_duration_s=0.6,
        min_median_conf=0.25,
        min_move_px=30.0,
        min_path_len_px=60.0,
        max_missing_xy_ratio=0.30,
        interp_limit=8,
        smooth_window=5,
        max_step_px=120.0,
        max_speed_px_s=2500.0,
        max_jump_ratio=0.02,
        max_tortuosity=3.0,
    )

    INPUT_PATH = "analysis_git200/data/full/all_tracks_test.csv"
    OUTPUT_DIR = "analysis_git200/data/processed_dataset"

    tidy, summary = build_tidy_dataset(
        csv_path=INPUT_PATH,
        out_dir=OUTPUT_DIR,
        cfg=cfg,
    )
    print("\nTop reject reasons:")
    print(summary.loc[~summary["keep"], "reject_reason"].value_counts().head(10))