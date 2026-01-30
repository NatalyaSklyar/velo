# analyze_tracks_csv.py
import csv
import sys
from collections import defaultdict
from typing import Dict, Any


def _to_float(s: str):
    try:
        return float(s)
    except Exception:
        return None


def analyze_csv(path: str):
    episodes: Dict[int, Dict[str, Any]] = {}
    total_rows = 0
    total_dets = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("[!] Empty CSV or missing header")
            return

        for row in reader:
            total_rows += 1
            ep = int(row["episode"])
            epd = episodes.get(ep)
            if epd is None:
                epd = {
                    "frames": 0,
                    "dets": 0,
                    "start_sec": None,
                    "end_sec": None,
                    "conf_sum": 0.0,
                    "conf_cnt": 0,
                    "raw_cnt": 0,
                    "filt_cnt": 0,
                }
                episodes[ep] = epd

            sec = _to_float(row.get("time", ""))
            if sec is not None:
                if epd["start_sec"] is None or sec < epd["start_sec"]:
                    epd["start_sec"] = sec
                if epd["end_sec"] is None or sec > epd["end_sec"]:
                    epd["end_sec"] = sec

            epd["frames"] += 1

            conf = _to_float(row.get("conf", ""))
            if conf is not None:
                epd["dets"] += 1
                total_dets += 1
                epd["conf_sum"] += conf
                epd["conf_cnt"] += 1

            raw_x = row.get("raw_x", "")
            raw_y = row.get("raw_y", "")
            if raw_x != "" and raw_y != "":
                epd["raw_cnt"] += 1

            filt_x = row.get("filt_x", "")
            filt_y = row.get("filt_y", "")
            if filt_x != "" and filt_y != "":
                epd["filt_cnt"] += 1

    print(f"[i] Rows: {total_rows}")
    print(f"[i] Episodes: {len(episodes)}")
    print(f"[i] Detections: {total_dets}")

    if not episodes:
        return

    print("\nPer-episode summary:")
    print("ep | start_s | end_s | dur_s | frames | dets | det_rate | mean_conf | raw_rate | filt_rate")
    for ep in sorted(episodes.keys()):
        d = episodes[ep]
        start_s = d["start_sec"] if d["start_sec"] is not None else 0.0
        end_s = d["end_sec"] if d["end_sec"] is not None else 0.0
        dur = max(0.0, end_s - start_s)
        frames = d["frames"]
        dets = d["dets"]
        det_rate = dets / frames if frames else 0.0
        mean_conf = (d["conf_sum"] / d["conf_cnt"]) if d["conf_cnt"] else 0.0
        raw_rate = d["raw_cnt"] / frames if frames else 0.0
        filt_rate = d["filt_cnt"] / frames if frames else 0.0

        print(
            f"{ep:2d} | {start_s:7.2f} | {end_s:6.2f} | {dur:5.2f} | {frames:6d} |"
            f" {dets:4d} | {det_rate:7.2f} | {mean_conf:9.3f} | {raw_rate:8.2f} | {filt_rate:9.2f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "all_tracks_test_18-21.csv"
    else:
        path = sys.argv[1]

    analyze_csv(path)
