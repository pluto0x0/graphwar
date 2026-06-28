from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    pixel_x: float
    pixel_y: float
    color: str


def find_board(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inner white board bounds as (left, top, right, bottom)."""
    # The grid contains thin dark lines. Closing reconnects the white cells while
    # preserving the much larger board rectangle.
    bright = cv2.inRange(image, (175, 175, 175), (255, 255, 255))
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
    )
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到白色坐标区域")

    height, width = image.shape[:2]
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Keep this low enough for full-desktop captures on multiple monitors.
        if w * h >= width * height * 0.08 and w / h >= 1.2:
            candidates.append((x, y, x + w - 1, y + h - 1))
    if not candidates:
        raise ValueError("未找到足够大的白色坐标区域")

    return max(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))


def _color_masks(hsv: np.ndarray) -> list[tuple[str, np.ndarray]]:
    # OpenCV hue range is [0, 179]. These deliberately broad ranges tolerate
    # antialiasing and modest changes in display colors.
    blue = cv2.inRange(hsv, (95, 100, 70), (140, 255, 255))
    # Player faces vary from orange-yellow to pale yellow depending on capture
    # and scaling, so use a wider range than the target-blue range.
    yellow = cv2.inRange(hsv, (10, 65, 90), (48, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return [
        ("blue", cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)),
        ("yellow", cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)),
    ]


def detect_points(
    image: np.ndarray, board: tuple[int, int, int, int] | None = None
) -> tuple[tuple[int, int, int, int], list[Detection]]:
    board = board or find_board(image)
    left, top, right, bottom = board
    if right <= left or bottom <= top:
        raise ValueError("坐标区域边界无效")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    board_width = right - left
    board_height = bottom - top
    scale = min(board_width, board_height)
    min_area = board_width * board_height * 0.00006
    max_area = board_width * board_height * 0.003
    raw: list[tuple[float, float, str, float]] = []

    for color, mask in _color_masks(hsv):
        roi_mask = np.zeros(mask.shape, dtype=np.uint8)
        roi_mask[top : bottom + 1, left : right + 1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if not min_area <= area <= max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h
            radius = (w + h) / 4
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
            min_aspect, max_aspect = (0.45, 1.80) if color == "yellow" else (0.65, 1.35)
            min_circularity = 0.25 if color == "yellow" else 0.55
            if not (
                min_aspect <= aspect <= max_aspect
                and 0.006 <= radius / scale <= 0.04
            ):
                continue
            # The green cap and facial details remove a significant part of the
            # yellow disk, so it must not be tested as a complete circle.
            if circularity < min_circularity:
                continue
            # An enclosing-circle center is less biased than the centroid when
            # the green player cap occludes part of the yellow face.
            (cx, cy), _ = cv2.minEnclosingCircle(contour)
            raw.append((cx, cy, color, area))

    # Merge nested or antialiased contours that describe the same dot.
    raw.sort(key=lambda item: item[3], reverse=True)
    kept: list[tuple[float, float, str, float]] = []
    for candidate in raw:
        if all(np.hypot(candidate[0] - p[0], candidate[1] - p[1]) > scale * 0.02 for p in kept):
            kept.append(candidate)

    detections = [
        Detection(
            x=round(-25 + 50 * (cx - left) / board_width, 3),
            y=round(15 - 30 * (cy - top) / board_height, 3),
            pixel_x=round(cx, 1),
            pixel_y=round(cy, 1),
            color=color,
        )
        for cx, cy, color, _ in kept
    ]
    detections.sort(key=lambda point: (point.x, -point.y))
    return board, detections


def draw_debug(
    image: np.ndarray,
    board: tuple[int, int, int, int],
    detections: list[Detection],
) -> np.ndarray:
    result = image.copy()
    left, top, right, bottom = board
    cv2.rectangle(result, (left, top), (right, bottom), (0, 180, 0), 2)
    for point in detections:
        center = (round(point.pixel_x), round(point.pixel_y))
        cv2.circle(result, center, 28, (0, 255, 0), 3)
        cv2.putText(
            result,
            f"({point.x:.2f}, {point.y:.2f})",
            (center[0] + 32, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 120, 0),
            2,
            cv2.LINE_AA,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="定位 Graphwar 截图中的玩家和目标")
    parser.add_argument("image", type=Path, help="截图路径")
    parser.add_argument("--debug-output", type=Path, help="保存带检测标记的图片")
    parser.add_argument(
        "--board",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="手动指定白色坐标区域的像素边界",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"无法读取图片: {args.image}")

    try:
        board, points = detect_points(image, tuple(args.board) if args.board else None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output = [{"x": point.x, "y": point.y} for point in points]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.debug_output:
        args.debug_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.debug_output), draw_debug(image, board, points)):
            raise SystemExit(f"无法写入调试图片: {args.debug_output}")


if __name__ == "__main__":
    main()
