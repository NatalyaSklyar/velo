import cv2
import numpy as np
from typing import Tuple, Optional

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


class MotionTriggerROI:
    """
    Дешёвый детектор движения в фиксированном ROI:
    - берёт фон как медиану первых N кадров (в ROI)
    - далее считает absdiff и долю изменившихся пикселей
    - если ratio >= motion_ratio K кадров подряд -> trigger
    - фон можно слегка обновлять EMA когда движения нет (anti-flicker по свету)
    """

    def __init__(
        self,
        W: int,
        H: int,
        fps: float,
        roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
        side_div: int = DEFAULT_SIDE_DIV,
        anchor: str = DEFAULT_ANCHOR,
        vertical_divisor: float = DEFAULT_VERTICAL_DIVISOR,
        diff_thresh: int = 25,
        motion_ratio: float = 0.02,
        consec_frames: int = 3,
        cooldown_sec: float = 2.0,
        bg_build_sec: float = 2.0,
        bg_update_alpha: float = 0.02,  # 0 отключает обновление
        bg_update_ratio_gate: float = 0.5,  # обновляем фон если ratio < motion_ratio*gate
        blur_ksize: int = 5,
    ):
        self.W = W
        self.H = H
        self.fps = float(fps) if fps and fps > 0 else 25.0
        if roi_xyxy is None:
            roi_xyxy = compute_sample_crop_xyxy_from_wh(
                W=W,
                H=H,
                side_div=side_div,
                anchor=anchor,
                vertical_divisor=vertical_divisor,
            )
        self._roi_xyxy = clamp_roi(*roi_xyxy, W, H)

        self.diff_thresh = int(diff_thresh)
        self.motion_ratio = float(motion_ratio)
        self.consec_frames = int(consec_frames)
        self.cooldown_frames = int(max(0, round(cooldown_sec * self.fps)))
        self.bg_build_frames = int(max(1, round(bg_build_sec * self.fps)))
        self.bg_update_alpha = float(bg_update_alpha)
        self.bg_update_ratio_gate = float(bg_update_ratio_gate)
        self.blur_ksize = int(blur_ksize) if blur_ksize % 2 == 1 else int(blur_ksize + 1)

        self.bg_roi: Optional[np.ndarray] = None
        self._consec = 0
        self._cooldown = 0

    def roi_xyxy(self) -> Tuple[int, int, int, int]:
        return self._roi_xyxy

    def build_background_from_cap(self, cap: cv2.VideoCapture):
        """Собирает фон (медиана) по ROI на первых bg_build_frames текущей позиции cap."""
        x1, y1, x2, y2 = self.roi_xyxy()
        samples = []
        pos0 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        for _ in range(self.bg_build_frames):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append(gray[y1:y2, x1:x2])

        if not samples:
            raise RuntimeError("Не удалось собрать фон для trigger (нет кадров).")

        self.bg_roi = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)

        # вернёмся обратно, чтобы пайплайн не “съедал” кадры
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)

    def build_background_from_image(
        self,
        image_path: str,
        crop_xyxy: Optional[Tuple[int, int, int, int]] = None,
        resize_to_roi: bool = True,
    ):
        """Берёт фон из заранее заданного изображения"""
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Не удалось прочитать sample image: {image_path}")

        Hs, Ws = img.shape[:2]
        if crop_xyxy is None:
            sx1, sy1, sx2, sy2 = 0, 0, Ws, Hs
        else:
            sx1, sy1, sx2, sy2 = clamp_roi(*crop_xyxy, Ws, Hs)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sample_roi = gray[sy1:sy2, sx1:sx2]

        rx1, ry1, rx2, ry2 = self.roi_xyxy()
        roi_w = rx2 - rx1
        roi_h = ry2 - ry1
        if (sample_roi.shape[1], sample_roi.shape[0]) != (roi_w, roi_h):
            if resize_to_roi:
                sample_roi = cv2.resize(sample_roi, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
            else:
                raise RuntimeError(
                    "Размер crop из sample не совпадает с ROI. "
                    "Включи resize_to_roi=True или подбери ROI."
                )

        self.bg_roi = sample_roi.copy()

    def build_background_from_sample_video(
        self,
        video_path: str,
        out_path: str,
        start_sec: float,
        crop_xyxy: Optional[Tuple[int, int, int, int]] = None,
        resize_to_roi: bool = True,
        side_div: int = DEFAULT_SIDE_DIV,
        anchor: str = DEFAULT_ANCHOR,
        vertical_divisor: float = DEFAULT_VERTICAL_DIVISOR,
    ):
        """Генерит sample image из видео и ставит фон из него."""
        from build_sample_image import build_sample_image

        crop_xyxy = build_sample_image(
            video_path=video_path,
            out_path=out_path,
            start_sec=start_sec,
            crop_xyxy=crop_xyxy,
            side_div=side_div,
            anchor=anchor,
            vertical_divisor=vertical_divisor,
        )
        self.build_background_from_image(
            image_path=out_path,
            crop_xyxy=crop_xyxy,
            resize_to_roi=resize_to_roi,
        )

    def _motion_ratio(self, roi_gray: np.ndarray) -> float:
        assert self.bg_roi is not None
        diff = cv2.absdiff(roi_gray, self.bg_roi)
        _, th = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        if self.blur_ksize >= 3:
            th = cv2.medianBlur(th, self.blur_ksize)
        changed = cv2.countNonZero(th)
        return changed / float(th.size)

    def step(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Возвращает (triggered, ratio)
        triggered=True только на кадре срабатывания (фронт).
        """
        if self.bg_roi is None:
            raise RuntimeError("bg_roi не инициализирован. Вызови build_background_from_cap().")

        x1, y1, x2, y2 = self.roi_xyxy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y1:y2, x1:x2]

        ratio = self._motion_ratio(roi_gray)

        # cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
            self._consec = 0
            # опционально: обновлять фон даже в кулдауне, если “тихо”
            if self.bg_update_alpha > 0 and ratio < self.motion_ratio * self.bg_update_ratio_gate:
                self.bg_roi = cv2.addWeighted(self.bg_roi, 1 - self.bg_update_alpha, roi_gray, self.bg_update_alpha, 0)
            return False, ratio

        # consec logic
        if ratio >= self.motion_ratio:
            self._consec += 1
        else:
            self._consec = 0
            # обновление фона в спокойном состоянии
            if self.bg_update_alpha > 0 and ratio < self.motion_ratio * self.bg_update_ratio_gate:
                self.bg_roi = cv2.addWeighted(self.bg_roi, 1 - self.bg_update_alpha, roi_gray, self.bg_update_alpha, 0)

        if self._consec >= self.consec_frames:
            self._consec = 0
            self._cooldown = self.cooldown_frames
            return True, ratio

        return False, ratio
