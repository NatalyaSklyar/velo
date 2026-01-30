import math

DEFAULT_SIDE_DIV = 3
DEFAULT_ANCHOR = "right"
DEFAULT_VERTICAL_DIVISOR = 1.5


def compute_sample_crop_xyxy_from_wh(
    W: int,
    H: int,
    side_div: int = DEFAULT_SIDE_DIV,
    anchor: str = DEFAULT_ANCHOR,
    vertical_divisor: float = DEFAULT_VERTICAL_DIVISOR,
) -> tuple[int, int, int, int]:
    # квадрат 1/side_div от изображения (берём от меньшей стороны, чтобы гарантированно влез)
    side = int(math.floor(min(W, H) / side_div))

    if anchor == "right":
        x1 = W - side
    else:
        x1 = 0

    # прижать по центру по вертикали (как было)
    y1 = int((H - side) / float(vertical_divisor))

    x2 = x1 + side
    y2 = y1 + side
    return x1, y1, x2, y2
