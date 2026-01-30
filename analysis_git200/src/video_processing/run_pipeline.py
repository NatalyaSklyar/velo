# run_pipeline.py
import cv2
from ultralytics import YOLO

from light_trigger import MotionTriggerROI
from sample_crop import DEFAULT_ANCHOR, DEFAULT_SIDE_DIV, DEFAULT_VERTICAL_DIVISOR
from tracker_segment import process_track_segment


def run_full_video_pipeline(
    video_path: str,
    model_path: str,
    out_csv_path: str = "all_tracks.csv",
    # time window (опционально)
    start_sec: float = 0.0,
    end_sec: float = None,
    # ROI/трек параметры (должны совпадать в trigger и track)
    side_div: int = DEFAULT_SIDE_DIV,
    roi_anchor: str = DEFAULT_ANCHOR,
    vertical_divisor: float = DEFAULT_VERTICAL_DIVISOR,
    # trigger bg from sample image (опционально)
    sample_image_path: str = None,
    sample_crop_xyxy: tuple = None,  # coords в системе полного кадра, если sample не обрезан
    sample_resize_to_roi: bool = True,
    # trigger params
    diff_thresh: int = 25,
    motion_ratio: float = 0.02,
    consec_frames: int = 3,
    cooldown_sec: float = 2.0,
    bg_build_sec: float = 2.0,
    # tracking params
    conf_thres: float = 0.75,
    # при триггере “отматываем” на 1 секунду
    rewind_sec: float = 1.0,
    # stop params сегмента
    stop_lost_patience: int = 20,
    max_track_seconds: float = 12.0,
    # video save (опционально, по умолчанию выключено)
    save_video: bool = False,
    out_video_path: str = "debug_full.mp4",
    draw_debug: bool = False,
    # trigger pass visualization/logging (опционально, по умолчанию выключено)
    save_trigger_video: bool = False,
    trigger_video_path: str = "debug_trigger.mp4",
    trigger_log_every: int = 0,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    model = YOLO(model_path)

    # writer (опционально, сегменты)
    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )
    # trigger writer (опционально, весь проход)
    trigger_writer = None
    if save_trigger_video:
        trigger_writer = cv2.VideoWriter(
            trigger_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

    roi_xyxy = sample_crop_xyxy

    trigger = MotionTriggerROI(
        W=W,
        H=H,
        fps=fps,
        roi_xyxy=roi_xyxy,
        side_div=side_div,
        anchor=roi_anchor,
        vertical_divisor=vertical_divisor,
        diff_thresh=diff_thresh,
        motion_ratio=motion_ratio,
        consec_frames=consec_frames,
        cooldown_sec=cooldown_sec,
        bg_build_sec=bg_build_sec,
    )

    start_frame = int(max(0, round(float(start_sec) * fps)))
    end_frame = None
    if end_sec is not None:
        end_frame = int(max(0, round(float(end_sec) * fps)))
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if sample_image_path is not None:
        trigger.build_background_from_image(
            image_path=sample_image_path,
            crop_xyxy=sample_crop_xyxy,
            resize_to_roi=sample_resize_to_roi,
        )
    else:
        trigger.build_background_from_cap(cap)

    # общий CSV
    with open(out_csv_path, "w", encoding="utf-8") as csv_f:
        csv_f.write(
            "episode,frame,time,raw_x,raw_y,filt_x,filt_y,conf,"
            "roi_x1,roi_y1,roi_x2,roi_y2,cx,cy,rw,rh\n"
        )

        episode_id = 0
        fired_total = 0
        processed_total = 0

        # идём по видео дешёвым проходом
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  
            cur_frame = max(0, frame_id - 1)
            processed_total += 1
            if end_frame is not None and cur_frame > end_frame:
                break

            fired, ratio = trigger.step(frame)

            if trigger_log_every and (processed_total % max(1, trigger_log_every) == 0):
                print(f"[TRIGGER] frame={cur_frame} sec={cur_frame / fps:.2f} ratio={ratio:.4f}")

            if trigger_writer is not None:
                x1, y1, x2, y2 = trigger.roi_xyxy()
                color = (0, 0, 255) if fired else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ratio:{ratio:.4f} fired:{int(fired)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
                trigger_writer.write(frame)

            if not fired:
                continue

            episode_id += 1
            fired_total += 1
            print(f"[TRIGGER] fired at frame={cur_frame} sec={cur_frame / fps:.2f} ratio={ratio:.4f}")

            # отмотка на rewind_sec
            rewind_frames = int(round(rewind_sec * fps))
            start_frame = max(0, cur_frame - rewind_frames)

            stats = process_track_segment(
                cap=cap,
                model=model,
                csv_f=csv_f,
                episode_id=episode_id,
                fps=fps,
                W=W,
                H=H,
                start_frame=start_frame,
                conf_thres=conf_thres,
                roi_xyxy=roi_xyxy,
                side_div=side_div,
                anchor=roi_anchor,
                vertical_divisor=vertical_divisor,
                stop_lost_patience=stop_lost_patience,
                max_track_seconds=max_track_seconds,
                writer=writer,
                draw_debug=draw_debug,
            )

            if sample_image_path is not None:
                trigger.build_background_from_image(
                    image_path=sample_image_path,
                    crop_xyxy=sample_crop_xyxy,
                    resize_to_roi=sample_resize_to_roi,
                )
            else:
                trigger.build_background_from_cap(cap)

            # краткая статистика
            print(
                f"[EP {episode_id}] start={stats['start_sec']:.2f}s end={stats['end_sec']:.2f}s "
                f"frames={stats['processed_frames']} dets={stats['detections']}"
            )

    cap.release()
    if writer is not None:
        writer.release()
    if trigger_writer is not None:
        trigger_writer.release()

    print(f"[✓] CSV сохранён: {out_csv_path}")
    if save_video:
        print(f"[✓] Debug video сохранено: {out_video_path}")
    if save_trigger_video:
        print(f"[✓] Trigger debug video сохранено: {trigger_video_path}")
    print(f"[i] Trigger fired: {fired_total}, frames processed in window: {processed_total}")


if __name__ == "__main__":
    VIDEO_PATH = "analysis_git200/data/comprec-full.mp4"
    MODEL_PATH = "runs/pose_cyclist/single_split/weights/best.pt"

    run_full_video_pipeline(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        out_csv_path="all_tracks.csv",

        # ROI
        roi_anchor="right",
        vertical_divisor=1.5,

        # trigger
        diff_thresh=25,
        motion_ratio=0.02,
        consec_frames=3,
        cooldown_sec=2.0,
        bg_build_sec=2.0,

        # track
        conf_thres=0.75,
        rewind_sec=1.0,
        stop_lost_patience=20,
        max_track_seconds=12.0,
    )
