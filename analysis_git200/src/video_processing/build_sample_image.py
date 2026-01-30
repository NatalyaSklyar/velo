import ffmpeg

from sample_crop import (
    DEFAULT_ANCHOR,
    DEFAULT_SIDE_DIV,
    DEFAULT_VERTICAL_DIVISOR,
    compute_sample_crop_xyxy_from_wh,
)

DEFAULT_VIDEO_PATH = "analysis_git200/data/comprec-full.mp4"
DEFAULT_OUT_PATH = "analysis_git200/data/sample.jpg"
DEFAULT_START_SEC = 18 * 60


def get_video_wh(path: str) -> tuple[int, int]:
    info = ffmpeg.probe(path)
    vstreams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise RuntimeError("No video stream found")
    s = vstreams[0]
    return int(s["width"]), int(s["height"])


def compute_sample_crop_xyxy(
    video_path: str,
    side_div: int = DEFAULT_SIDE_DIV,
    anchor: str = DEFAULT_ANCHOR,
    vertical_divisor: float = DEFAULT_VERTICAL_DIVISOR,
) -> tuple[int, int, int, int]:
    W, H = get_video_wh(video_path)
    return compute_sample_crop_xyxy_from_wh(
        W=W,
        H=H,
        side_div=side_div,
        anchor=anchor,
        vertical_divisor=vertical_divisor,
    )


def build_sample_image(
    video_path: str = DEFAULT_VIDEO_PATH,
    out_path: str = DEFAULT_OUT_PATH,
    start_sec: float = DEFAULT_START_SEC,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    side_div: int = DEFAULT_SIDE_DIV,
    anchor: str = DEFAULT_ANCHOR,
    vertical_divisor: float = DEFAULT_VERTICAL_DIVISOR,
) -> tuple[int, int, int, int]:
    if crop_xyxy is None:
        crop_xyxy = compute_sample_crop_xyxy(
            video_path=video_path,
            side_div=side_div,
            anchor=anchor,
            vertical_divisor=vertical_divisor,
        )

    x1, y1, x2, y2 = crop_xyxy
    w = x2 - x1
    h = y2 - y1

    (
        ffmpeg
        .input(video_path, ss=start_sec)
        .filter("crop", x=x1, y=y1, w=w, h=h)
        .output(out_path, vframes=1)
        .overwrite_output()
        .run()
    )

    return crop_xyxy


if __name__ == "__main__":
    crop = build_sample_image()
    W, H = get_video_wh(DEFAULT_VIDEO_PATH)
    print("Video WH:", (W, H))
    print("SAMPLE_CROP_XYXY:", crop)
