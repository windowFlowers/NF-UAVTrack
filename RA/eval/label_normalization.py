"""Strict label and prediction normalization for all evaluation datasets.

The internal representation is always ``xywh`` in image pixels, one row per
video frame. Missing annotations are represented by four NaNs. Input formats
must be declared when they are ambiguous; the loader never silently guesses a
five-column annotation file.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


ABSENT_XYWH = np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_MISSING_TOKENS = {"", "nan", "na", "none", "null", "-", "-1.#ind"}


class LabelFormatError(ValueError):
    """Raised when an annotation cannot be normalized without guessing."""


@dataclass(frozen=True)
class LabelReport:
    path: str
    format: str
    rows: int
    absent_rows: int
    valid_rows: int
    frame_count_expected: Optional[int]
    frame_size: Optional[tuple[int, int]]


def load_xywh_labels(
    path: str | Path,
    *,
    format: str = "xywh_pixel",
    frame_size: Optional[Sequence[int]] = None,
    expected_frames: Optional[int] = None,
    strict: bool = True,
) -> np.ndarray:
    """Load one annotation file into a float32 ``[N, 4]`` xywh array.

    Supported formats are ``xywh_pixel`` (four columns),
    ``xywh_pixel_with_frame_id`` (frame id plus four columns),
    ``yolo_normalized`` (class, cx, cy, w, h), and ``prediction`` (four or
    five columns, with confidence ignored in the returned array).
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")
    rows = []
    frame_ids = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            if strict and format == "yolo_normalized":
                raise LabelFormatError(f"Blank YOLO label line at {path}:{line_number}")
            continue
        tokens = _split_tokens(line)
        try:
            parsed, frame_id = _parse_row(tokens, format, frame_size, path, line_number)
        except LabelFormatError:
            raise
        except (TypeError, ValueError) as exc:
            raise LabelFormatError(f"Invalid numeric label at {path}:{line_number}: {line}") from exc
        rows.append(parsed)
        if frame_id is not None:
            frame_ids.append(frame_id)

    array = np.asarray(rows, dtype=np.float32).reshape((-1, 4)) if rows else np.empty((0, 4), dtype=np.float32)
    if frame_ids:
        _validate_frame_ids(frame_ids, path)
    return _enforce_frame_count(array, expected_frames, path, strict=strict)


def load_prediction_file(
    path: str | Path,
    *,
    expected_frames: Optional[int] = None,
    strict: bool = True,
) -> np.ndarray:
    """Load project prediction rows as ``[N, 5]`` xywh plus confidence."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
    rows = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = _split_tokens(line)
        if len(tokens) not in (4, 5):
            raise LabelFormatError(
                f"Prediction row at {path}:{line_number} must have 4 or 5 columns, got {len(tokens)}"
            )
        values = [_parse_float(token, path, line_number) for token in tokens]
        box = np.asarray(values[:4], dtype=np.float32)
        confidence = float(values[4]) if len(values) == 5 else 1.0
        if _is_absent_box(box):
            rows.append([np.nan, np.nan, np.nan, np.nan, 0.0])
            continue
        _validate_box(box, path, line_number, allow_outside=True)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise LabelFormatError(f"Invalid confidence at {path}:{line_number}: {confidence}")
        rows.append([*box.tolist(), confidence])
    array = np.asarray(rows, dtype=np.float32).reshape((-1, 5)) if rows else np.empty((0, 5), dtype=np.float32)
    if expected_frames is not None and len(array) != int(expected_frames):
        raise LabelFormatError(
            f"Frame count mismatch for {path}: expected {int(expected_frames)}, got {len(array)}"
        )
    return array


def load_yolo_frame_directory(
    directory: str | Path,
    *,
    frame_count: int,
    frame_size: Sequence[int],
    strict: bool = True,
) -> np.ndarray:
    """Load one YOLO TXT file per frame into pixel-space xywh labels.

    Empty files are valid absent frames. File names must contain a unique
    integer frame index; indices may be zero- or one-based but must be
    consecutive after sorting.
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"YOLO label directory does not exist: {directory}")
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
        key=_frame_file_sort_key,
    )
    expected = int(frame_count)
    if len(files) != expected:
        raise LabelFormatError(
            f"Frame label file count mismatch for {directory}: expected {expected}, got {len(files)}"
        )
    indices = [_frame_file_sort_key(path)[0] for path in files]
    if len(set(indices)) != len(indices):
        raise LabelFormatError(f"Duplicate frame label indices in {directory}: {indices[:8]}")
    if indices and indices != list(range(indices[0], indices[0] + expected)):
        raise LabelFormatError(
            f"Non-consecutive frame label indices in {directory}: {indices[:8]}"
        )
    rows = []
    for path in files:
        if not path.read_text(encoding="utf-8").strip():
            rows.append(ABSENT_XYWH.copy())
            continue
        parsed = load_xywh_labels(
            path,
            format="yolo_normalized",
            frame_size=frame_size,
            expected_frames=1,
            strict=strict,
        )
        if len(parsed) != 1:
            raise LabelFormatError(f"Expected one target row in {path}, got {len(parsed)}")
        rows.append(parsed[0])
    return np.asarray(rows, dtype=np.float32)


def load_sequence_labels(
    sequence_dir: str | Path,
    *,
    frame_count: Optional[int] = None,
    frame_size: Optional[Sequence[int]] = None,
    format: str = "xywh_pixel",
    yolo_label_dir: Optional[str | Path] = None,
    strict: bool = True,
) -> np.ndarray:
    """Load labels from a sequence directory using one explicit format."""

    sequence_dir = Path(sequence_dir)
    if yolo_label_dir is not None:
        if frame_count is None or frame_size is None:
            raise LabelFormatError("YOLO frame labels require frame_count and frame_size")
        return load_yolo_frame_directory(
            yolo_label_dir,
            frame_count=int(frame_count),
            frame_size=frame_size,
            strict=strict,
        )
    path = sequence_dir / "groundtruth.txt"
    return load_xywh_labels(
        path,
        format=format,
        frame_size=frame_size,
        expected_frames=frame_count,
        strict=strict,
    )


def validate_sequence_labels(
    sequence_dir: str | Path,
    *,
    frame_count: int,
    frame_size: Sequence[int],
    format: str = "xywh_pixel",
    yolo_label_dir: Optional[str | Path] = None,
) -> LabelReport:
    """Validate one sequence and return a machine-readable label report."""

    labels = load_sequence_labels(
        sequence_dir,
        frame_count=frame_count,
        frame_size=frame_size,
        format=format,
        yolo_label_dir=yolo_label_dir,
        strict=True,
    )
    valid = _valid_box_mask(labels)
    return LabelReport(
        path=str(yolo_label_dir or (Path(sequence_dir) / "groundtruth.txt")),
        format="yolo_normalized" if yolo_label_dir is not None else format,
        rows=int(len(labels)),
        absent_rows=int((~valid).sum()),
        valid_rows=int(valid.sum()),
        frame_count_expected=int(frame_count),
        frame_size=(int(frame_size[0]), int(frame_size[1])),
    )


def normalize_prediction_rows(rows: Iterable[Sequence[float]]) -> np.ndarray:
    """Normalize tracker output to a finite ``[N, 5]`` prediction array."""

    normalized = []
    for index, row in enumerate(rows):
        values = list(row)
        if len(values) not in (4, 5):
            raise LabelFormatError(f"Prediction row {index} must have 4 or 5 values")
        box = np.asarray(values[:4], dtype=np.float32)
        score = float(values[4]) if len(values) == 5 else 1.0
        if _is_absent_box(box) or not _valid_box_mask(box.reshape(1, 4))[0]:
            normalized.append([np.nan, np.nan, np.nan, np.nan, 0.0])
        else:
            _validate_box(box, Path("<prediction>"), index + 1, allow_outside=True)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise LabelFormatError(f"Prediction row {index} has invalid confidence {score}")
            normalized.append([*box.tolist(), score])
    return np.asarray(normalized, dtype=np.float32).reshape((-1, 5))


def write_prediction_file(path: str | Path, rows: Iterable[Sequence[float]]) -> None:
    """Write canonical ``x,y,w,h,confidence`` rows with no silent row drops."""

    array = normalize_prediction_rows(rows)
    lines = []
    for row in array:
        if not _valid_box_mask(row[:4].reshape(1, 4))[0]:
            lines.append("nan,nan,nan,nan,0.000000")
        else:
            lines.append(",".join(f"{float(value):.6f}" for value in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _parse_row(tokens, format_name, frame_size, path, line_number):
    if format_name == "xywh_pixel":
        if len(tokens) != 4:
            raise LabelFormatError(f"Expected 4 columns at {path}:{line_number}, got {len(tokens)}")
        values = np.asarray([_parse_float(token, path, line_number) for token in tokens], dtype=np.float32)
        if _is_absent_box(values):
            return ABSENT_XYWH.copy(), None
        _validate_box(values, path, line_number, allow_outside=True)
        return values, None
    if format_name == "xywh_pixel_with_frame_id":
        if len(tokens) != 5:
            raise LabelFormatError(f"Expected 5 columns at {path}:{line_number}, got {len(tokens)}")
        frame_id = _parse_frame_id(tokens[0], path, line_number)
        values = np.asarray([_parse_float(token, path, line_number) for token in tokens[1:]], dtype=np.float32)
        if _is_absent_box(values):
            return ABSENT_XYWH.copy(), frame_id
        _validate_box(values, path, line_number, allow_outside=True)
        return values, frame_id
    if format_name == "yolo_normalized":
        if len(tokens) != 5:
            raise LabelFormatError(f"Expected class,cx,cy,w,h at {path}:{line_number}, got {len(tokens)}")
        if frame_size is None or len(frame_size) != 2:
            raise LabelFormatError("yolo_normalized requires frame_size=(height,width)")
        class_id = _parse_float(tokens[0], path, line_number)
        if not math.isfinite(class_id) or class_id < 0 or not class_id.is_integer():
            raise LabelFormatError(f"Invalid YOLO class id at {path}:{line_number}")
        cx, cy, width, height = [_parse_float(token, path, line_number) for token in tokens[1:]]
        if not all(math.isfinite(value) for value in (cx, cy, width, height)):
            return ABSENT_XYWH.copy(), None
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise LabelFormatError(f"YOLO normalized values out of range at {path}:{line_number}")
        image_h, image_w = int(frame_size[0]), int(frame_size[1])
        values = np.asarray(
            [
                (cx - width / 2.0) * image_w,
                (cy - height / 2.0) * image_h,
                width * image_w,
                height * image_h,
            ],
            dtype=np.float32,
        )
        _validate_box(values, path, line_number, allow_outside=True)
        return values, None
    if format_name == "prediction":
        if len(tokens) not in (4, 5):
            raise LabelFormatError(f"Prediction rows require 4 or 5 columns at {path}:{line_number}")
        values = np.asarray([_parse_float(token, path, line_number) for token in tokens[:4]], dtype=np.float32)
        if _is_absent_box(values):
            return ABSENT_XYWH.copy(), None
        _validate_box(values, path, line_number, allow_outside=True)
        return values, None
    raise LabelFormatError(f"Unsupported label format: {format_name}")


def _split_tokens(line: str):
    return [token for token in re.split(r"[;,\s]+", line.strip()) if token]


def _parse_float(token, path, line_number):
    normalized = str(token).strip().lower()
    if normalized in _MISSING_TOKENS:
        return float("nan")
    try:
        return float(token)
    except (TypeError, ValueError) as exc:
        raise LabelFormatError(f"Non-numeric token at {path}:{line_number}: {token}") from exc


def _parse_frame_id(token, path, line_number):
    value = _parse_float(token, path, line_number)
    if not math.isfinite(value) or not value.is_integer() or value < 0:
        raise LabelFormatError(f"Invalid frame id at {path}:{line_number}: {token}")
    return int(value)


def _validate_frame_ids(frame_ids, path):
    if len(set(frame_ids)) != len(frame_ids):
        raise LabelFormatError(f"Duplicate frame ids in {path}")
    expected = list(range(frame_ids[0], frame_ids[0] + len(frame_ids)))
    if frame_ids != expected:
        raise LabelFormatError(f"Non-consecutive frame ids in {path}: {frame_ids[:5]}")


def _enforce_frame_count(array, expected_frames, path, strict):
    if expected_frames is not None and len(array) != int(expected_frames):
        message = f"Frame count mismatch for {path}: expected {int(expected_frames)}, got {len(array)}"
        if strict:
            raise LabelFormatError(message)
    return array


def _validate_box(box, path, line_number, allow_outside):
    if not np.isfinite(box).all():
        raise LabelFormatError(f"Partially missing bbox at {path}:{line_number}")
    if float(box[2]) <= 0.0 or float(box[3]) <= 0.0:
        raise LabelFormatError(f"Non-positive bbox size at {path}:{line_number}: {box.tolist()}")
    if not allow_outside and (float(box[0]) < 0.0 or float(box[1]) < 0.0):
        raise LabelFormatError(f"Bbox outside image at {path}:{line_number}")


def _is_absent_box(box):
    values = np.asarray(box, dtype=np.float32).reshape(4)
    return bool(
        np.isnan(values).all()
        or np.all(values == 0.0)
        or (np.isfinite(values).all() and np.all(values <= 0.0))
    )


def _valid_box_mask(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    return np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)


def _frame_file_sort_key(path: Path):
    return frame_path_sort_key(path)


def frame_path_sort_key(path: Path):
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise LabelFormatError(f"Frame label filename has no index: {path}")
    return int(matches[-1]), path.name


__all__ = [
    "ABSENT_XYWH",
    "LabelFormatError",
    "LabelReport",
    "load_prediction_file",
    "load_sequence_labels",
    "load_xywh_labels",
    "load_yolo_frame_directory",
    "normalize_prediction_rows",
    "frame_path_sort_key",
    "validate_sequence_labels",
    "write_prediction_file",
]
