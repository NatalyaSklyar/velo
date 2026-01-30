# tracker_segment.py
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from sample_crop import (
    DEFAULT_ANCHOR,
    DEFAULT_SIDE_DIV,
    DEFAULT_VERTICAL_DIVISOR,
    compute_sample_crop_xyxy_from_wh,
)


def clamp_roi(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(1, min(x2, W)))
    y2 = int(max(1, min(y2, H)))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def pick_target(results, prev_center=None, conf_thres=0.6, dist_weight=0.002):
    if results.keypoints is None or results.boxes is None:
        return None

    kpts = results.keypoints.xy.cpu().numpy()   # (N, K, 2) ROI coords
    confs = results.boxes.conf.cpu().numpy()    # (N,)
    boxes = results.boxes.xyxy.cpu().numpy()    # (N, 4) ROI coords

    best_i = None
    best_score = -1e9

    for i, (kp, conf, box) in enumerate(zip(kpts, confs, boxes)):
        if conf < conf_thres:
            continue

        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])

        score = float(conf)
        if prev_center is not None:
            px, py = prev_center
            d = np.hypot(cx - px, cy - py)
            score -= dist_weight * d

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None

    box = boxes[best_i]
    return {
        "i": best_i,
        "kp": kpts[best_i],
        "conf": float(confs[best_i]),
        "box": box,
        "center": (float(0.5 * (box[0] + box[2])), float(0.5 * (box[1] + box[3])))
    }


def make_kalman_2d(dt: float):
    # state [x, y, vx, vy], measurement [x, y]
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf


def process_track_segment(
    cap: cv2.VideoCapture,
    model,
    csv_f,
    episode_id: int,
    fps: float,
    W: int,
    H: int,
    start_frame: int,
    conf_thres: float = 0.75,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    side_div: int = DEFAULT_SIDE_DIV,
    anchor: str = DEFAULT_ANCHOR,
    vertical_divisor: float = DEFAULT_VERTICAL_DIVISOR,
    smooth: float = 0.25,
    lost_patience: int = 10,
    reacquire_grow: float = 1.15,
    max_grow: float = 2.0,
    stop_lost_patience: int = 20,
    max_track_seconds: float = 12.0,
    # keypoints midpoint 
    KP_A: int = 2,
    KP_B: int = 5,

    writer: Optional[cv2.VideoWriter] = None,
    draw_debug: bool = False,
) -> Dict[str, Any]:
    """
    Запускает "тяжёлую" обработку с YOLO pose + ROI слежением начиная с start_frame.
    Останавливается, когда объект потерян stop_lost_patience кадров подряд
    ИЛИ превышен max_track_seconds.

    Пишет строки в общий csv_f (один файл на всё видео).

    Возвращает dict со статистикой и last_frame (последний обработанный кадр).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(start_frame, 0))
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if roi_xyxy is None:
        roi_xyxy = compute_sample_crop_xyxy_from_wh(
            W=W,
            H=H,
            side_div=side_div,
            anchor=anchor,
            vertical_divisor=vertical_divisor,
        )
    x1, y1, x2, y2 = clamp_roi(*roi_xyxy, W, H)
    base_rw = x2 - x1
    base_rh = y2 - y1
    cx = x1 + base_rw // 2
    cy = y1 + base_rh // 2
    rw, rh = base_rw, base_rh

    prev_center_roi = None
    lost = 0

    dt = 1.0 / max(fps, 1e-6)
    kf = make_kalman_2d(dt)
    kf_inited = False

    # сегментная статистика
    start_sec = frame_id / fps
    max_frames = int(max_track_seconds * fps)
    processed = 0
    detections = 0

    # csv: episode, frame, time, raw_x, raw_y, filt_x, filt_y, conf, roi_x1, roi_y1, roi_x2, roi_y2, cx, cy, rw, rh

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        processed += 1
        sec = frame_id / fps

        # stop по времени сегмента
        if processed > max_frames:
            break

        # ROI around (cx, cy)
        x1 = cx - rw // 2
        y1 = cy - rh // 2
        x2 = cx + rw // 2
        y2 = cy + rh // 2
        x1, y1, x2, y2 = clamp_roi(x1, y1, x2, y2, W, H)

        roi = frame[y1:y2, x1:x2]
        results = model(roi, verbose=False)[0]
        target = pick_target(results, prev_center=prev_center_roi, conf_thres=conf_thres)

        raw_meas = None
        filt_xy = (np.nan, np.nan)
        conf_out = ""

        if target is not None:
            detections += 1
            conf_out = f"{target['conf']:.4f}"
            lost = 0

            tx, ty = target["center"]
            desired_cx = x1 + tx
            desired_cy = y1 + ty

            cx = int((1 - smooth) * cx + smooth * desired_cx)
            cy = int((1 - smooth) * cy + smooth * desired_cy)

            rw = int((1 - smooth) * rw + smooth * base_rw)
            rh = int((1 - smooth) * rh + smooth * base_rh)

            prev_center_roi = (tx, ty)

            kp = target["kp"]

            # RAW midpoint (KP_A, KP_B)
            if kp.shape[0] > max(KP_A, KP_B):
                xA, yA = kp[KP_A]
                xB, yB = kp[KP_B]
                if xA > 0 and yA > 0 and xB > 0 and yB > 0:
                    mx = 0.5 * (xA + xB)
                    my = 0.5 * (yA + yB)
                    raw_meas = (float(x1 + mx), float(y1 + my))

            # Kalman
            pred = kf.predict()
            if raw_meas is not None:
                if not kf_inited:
                    kf.statePost = np.array([[raw_meas[0]], [raw_meas[1]], [0.0], [0.0]], dtype=np.float32)
                    kf_inited = True
                    fx, fy = raw_meas
                else:
                    m = np.array([[raw_meas[0]], [raw_meas[1]]], dtype=np.float32)
                    est = kf.correct(m)
                    fx, fy = float(est[0]), float(est[1])
            else:
                if kf_inited:
                    fx, fy = float(pred[0]), float(pred[1])
                else:
                    fx, fy = np.nan, np.nan

            filt_xy = (fx, fy)

            if draw_debug:
                # keypoints
                for x, y in kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x1 + x), int(y1 + y)), 4, (0, 255, 0), -1)

                # bbox
                bx1, by1, bx2, by2 = target["box"]
                cv2.rectangle(frame, (int(x1 + bx1), int(y1 + by1)), (int(x1 + bx2), int(y1 + by2)), (255, 255, 0), 2)

                # raw
                if raw_meas is not None:
                    cv2.circle(frame, (int(raw_meas[0]), int(raw_meas[1])), 5, (0, 255, 255), -1)

                # kalman
                if kf_inited and not np.isnan(fx) and not np.isnan(fy):
                    cv2.circle(frame, (int(fx), int(fy)), 3, (0, 0, 255), -1)

        else:
            lost += 1
            rw = int(base_rw)
            rh = int(base_rh)
            if lost > lost_patience:
                prev_center_roi = None

            # только по калману 
            pred = kf.predict()
            if kf_inited:
                fx, fy = float(pred[0]), float(pred[1])
                filt_xy = (fx, fy)

        if lost >= stop_lost_patience:
            break

        # CSV
        raw_x = "" if raw_meas is None else f"{raw_meas[0]:.3f}"
        raw_y = "" if raw_meas is None else f"{raw_meas[1]:.3f}"
        fx, fy = filt_xy
        filt_x = "" if (not kf_inited or np.isnan(fx)) else f"{fx:.3f}"
        filt_y = "" if (not kf_inited or np.isnan(fy)) else f"{fy:.3f}"

        csv_f.write(
            f"{episode_id},{frame_id},{sec:.3f},{raw_x},{raw_y},{filt_x},{filt_y},{conf_out},"
            f"{x1},{y1},{x2},{y2},{cx},{cy},{rw},{rh}\n"
        )

        if writer is not None:
            if draw_debug:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ep:{episode_id} time:{sec:.2f}s lost:{lost}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            writer.write(frame)

    end_sec = frame_id / fps
    return {
        "episode_id": episode_id,
        "start_frame": start_frame,
        "start_sec": start_sec,
        "end_frame": frame_id,
        "end_sec": end_sec,
        "processed_frames": processed,
        "detections": detections,
        "last_frame": frame_id,
    }
