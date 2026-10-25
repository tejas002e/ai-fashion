"""Deterministic portrait validation and color-estimation helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from io import BytesIO
import warnings

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from sklearn.cluster import KMeans


MAX_UPLOAD_BYTES = 10_000_000
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 2_400
MAX_DETECTION_SIDE = 1_600
MIN_IMAGE_SIDE = 128
MIN_FACE_SIDE = 80


class PortraitAnalysisError(ValueError):
    """An expected validation or portrait-analysis failure."""


def decode_uploaded_image(data: bytes) -> Image.Image:
    """Validate, orient, normalize, and bound a JPEG or PNG portrait."""
    if not data:
        raise PortraitAnalysisError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise PortraitAnalysisError("The image is larger than the 10 MB upload limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image_format = image.format
                image.verify()

            if image_format not in {"JPEG", "PNG"}:
                raise PortraitAnalysisError(
                    "This file is not a supported JPEG or PNG image."
                )

            with Image.open(BytesIO(data)) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
                    raise PortraitAnalysisError(
                        f"The image is too small ({width}×{height}). "
                        f"Use an image at least {MIN_IMAGE_SIDE}×{MIN_IMAGE_SIDE} pixels."
                    )
                if width * height > MAX_IMAGE_PIXELS:
                    raise PortraitAnalysisError(
                        "The image dimensions are too large to analyze safely. "
                        "Use an image under 20 megapixels."
                    )

                image.load()
                has_transparency = (
                    "A" in image.getbands() or "transparency" in image.info
                )
                icc_profile = image.info.get("icc_profile")
                if has_transparency:
                    rgba = image.convert("RGBA")
                    rgb = rgba.convert("RGB")
                else:
                    rgb = image.convert("RGB")

                if icc_profile:
                    try:
                        input_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                        srgb_profile = ImageCms.createProfile("sRGB")
                        if image.mode == "CMYK" and not has_transparency:
                            rgb = ImageCms.profileToProfile(
                                image,
                                input_profile,
                                srgb_profile,
                                outputMode="RGB",
                            )
                        else:
                            rgb = ImageCms.profileToProfile(
                                rgb,
                                input_profile,
                                srgb_profile,
                                outputMode="RGB",
                            )
                    except (ImageCms.PyCMSError, OSError, ValueError) as exc:
                        raise PortraitAnalysisError(
                            "The image has an unreadable color profile. "
                            "Re-save it as an sRGB JPEG or PNG and try again."
                        ) from exc

                if has_transparency:
                    rgba_srgb = rgb.convert("RGBA")
                    rgba_srgb.putalpha(rgba.getchannel("A"))
                    background = Image.new(
                        "RGBA",
                        rgba_srgb.size,
                        (255, 255, 255, 255),
                    )
                    normalized = Image.alpha_composite(
                        background,
                        rgba_srgb,
                    ).convert("RGB")
                else:
                    normalized = rgb

                if max(normalized.size) > MAX_IMAGE_SIDE:
                    normalized.thumbnail(
                        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
                        Image.Resampling.LANCZOS,
                    )
                return normalized.copy()
    except PortraitAnalysisError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise PortraitAnalysisError(
            "The image is too large to decode safely. Choose a smaller image."
        ) from exc
    except (UnidentifiedImageError, OSError, EOFError, SyntaxError, ValueError) as exc:
        raise PortraitAnalysisError(
            "The image could not be decoded. It may be corrupted or unsupported."
        ) from exc


def select_largest_face(
    faces: list[tuple[int, int, int, int]] | np.ndarray,
    image_width: int,
    image_height: int,
    min_face_side: int = MIN_FACE_SIDE,
) -> tuple[
    tuple[int, int, int, int] | None,
    list[tuple[int, int, int, int]],
    bool,
]:
    """Return the largest usable face; ties resolve topmost, then leftmost."""
    candidates: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for face in faces:
        try:
            x, y, width, height = (int(value) for value in face)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(image_width, x + width)
        y1 = min(image_height, y + height)
        if x1 <= x0 or y1 <= y0:
            continue

        box = (x0, y0, x1 - x0, y1 - y0)
        if box not in seen:
            candidates.append(box)
            seen.add(box)

    usable = [
        box
        for box in candidates
        if box[2] >= min_face_side and box[3] >= min_face_side
    ]
    usable.sort(key=lambda box: (-(box[2] * box[3]), box[1], box[0], -box[3], -box[2]))
    selected = usable[0] if usable else None
    only_small_faces = bool(candidates) and not usable
    return selected, usable, only_small_faces


@lru_cache(maxsize=1)
def _face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise PortraitAnalysisError("The face detector could not be loaded.")
    return detector


def detect_face(
    image_rgb: np.ndarray,
) -> tuple[
    tuple[int, int, int, int] | None,
    list[tuple[int, int, int, int]],
    bool,
]:
    """Detect faces at bounded resolution and choose one reproducibly."""
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.uint8
    ):
        raise PortraitAnalysisError("The image is not in a valid RGB format.")

    image_height, image_width = image_rgb.shape[:2]
    if image_width == 0 or image_height == 0:
        raise PortraitAnalysisError("The image has no usable pixels.")

    scale = min(1.0, MAX_DETECTION_SIDE / max(image_width, image_height))
    if scale < 1.0:
        detection_width = max(1, round(image_width * scale))
        detection_height = max(1, round(image_height * scale))
        detection_image = cv2.resize(
            image_rgb,
            (detection_width, detection_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_image = image_rgb
        detection_height, detection_width = image_height, image_width

    gray = cv2.cvtColor(detection_image, cv2.COLOR_RGB2GRAY)
    detected = _face_detector().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    scale_x = image_width / detection_width
    scale_y = image_height / detection_height
    original_scale_boxes = [
        (
            round(int(x) * scale_x),
            round(int(y) * scale_y),
            round(int(width) * scale_x),
            round(int(height) * scale_y),
        )
        for x, y, width, height in detected
    ]
    return select_largest_face(
        original_scale_boxes,
        image_width,
        image_height,
    )


def crop_face(
    image_rgb: np.ndarray,
    face_rect: tuple[int, int, int, int],
) -> np.ndarray:
    """Validate a face box and return a non-empty crop."""
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.uint8
    ):
        raise PortraitAnalysisError("The image is not in a valid RGB format.")
    if not isinstance(face_rect, (tuple, list)) or len(face_rect) != 4:
        raise PortraitAnalysisError("The detected face region is invalid.")

    x, y, width, height = (int(value) for value in face_rect)
    image_height, image_width = image_rgb.shape[:2]
    if (
        x < 0
        or y < 0
        or width < MIN_FACE_SIDE
        or height < MIN_FACE_SIDE
        or x + width > image_width
        or y + height > image_height
    ):
        raise PortraitAnalysisError(
            "The detected face region is too small or outside the image. "
            "Try a closer, well-framed portrait."
        )

    face_roi = image_rgb[y : y + height, x : x + width]
    if face_roi.size == 0 or min(face_roi.shape[:2]) < MIN_FACE_SIDE:
        raise PortraitAnalysisError("The detected face crop is too small to analyze.")
    return face_roi


def extract_skin_color(face_roi: np.ndarray) -> tuple[tuple[int, int, int], int, int]:
    """Estimate color from separate forehead and cheek patches, not one center crop."""
    if (
        not isinstance(face_roi, np.ndarray)
        or face_roi.ndim != 3
        or face_roi.shape[2] != 3
        or face_roi.dtype != np.uint8
    ):
        raise PortraitAnalysisError("The detected face crop is invalid.")

    height, width = face_roi.shape[:2]
    if min(width, height) < MIN_FACE_SIDE:
        raise PortraitAnalysisError(
            "The detected face is too small for reliable color sampling."
        )

    # Normalized interior patches avoid the eyes, eyebrows, mouth, hairline,
    # face outline, and background while sampling both cheeks and the forehead.
    patches = (
        (0.32, 0.19, 0.68, 0.33),  # forehead
        (0.16, 0.47, 0.39, 0.65),  # image-left cheek
        (0.61, 0.47, 0.84, 0.65),  # image-right cheek
    )
    samples_by_patch: list[np.ndarray] = []
    max_samples_per_patch = 4_000

    for left, top, right, bottom in patches:
        x0 = max(0, round(width * left))
        x1 = min(width, round(width * right))
        y0 = max(0, round(height * top))
        y1 = min(height, round(height * bottom))
        pixels = face_roi[y0:y1, x0:x1].reshape(-1, 3)
        if pixels.size == 0:
            continue
        if len(pixels) > max_samples_per_patch:
            indices = np.linspace(
                0,
                len(pixels) - 1,
                num=max_samples_per_patch,
                dtype=np.int64,
            )
            pixels = pixels[indices]
        samples_by_patch.append(pixels)

    if not samples_by_patch:
        raise PortraitAnalysisError(
            "There were not enough usable forehead and cheek pixels to estimate color."
        )

    samples = np.concatenate(samples_by_patch, axis=0)
    if len(samples) < 30:
        raise PortraitAnalysisError(
            "There were not enough usable facial pixels to estimate color."
        )

    # Cluster in perceptual Lab space, then use a median RGB value from the
    # largest cluster so a few shadows or highlights have less influence.
    rgb_float = samples.astype(np.float32).reshape(-1, 1, 3) / 255.0
    lab_samples = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    unique_colors = np.unique(lab_samples, axis=0).shape[0]

    if unique_colors <= 1:
        dominant_cluster = samples
    else:
        cluster_count = min(3, unique_colors)
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
            max_iter=100,
        ).fit_predict(lab_samples)
        dominant_label = int(np.argmax(np.bincount(labels)))
        dominant_cluster = samples[labels == dominant_label]

    representative_rgb = np.rint(np.median(dominant_cluster, axis=0)).astype(int)
    color = tuple(int(channel) for channel in representative_rgb)
    return color, len(dominant_cluster), len(samples)


def classify_skin_color(
    rgb_color: tuple[int, int, int] | list[int] | np.ndarray,
) -> dict[str, str | tuple[float, float, float]]:
    """Return documented, display-only estimates from a representative RGB color."""
    rgb = np.asarray(rgb_color, dtype=np.float32)
    if (
        rgb.shape != (3,)
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0)
        or np.any(rgb > 255)
    ):
        raise PortraitAnalysisError("The measured RGB color is invalid.")

    rgb_float = (rgb / 255.0).reshape(1, 1, 3)
    lightness, a_axis, b_axis = (
        float(value)
        for value in cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)[0, 0]
    )

    # These broad CIELAB L* bins are presentation categories, not a universal
    # or validated skin-tone scale.
    if lightness >= 68:
        tone = "Light"
    elif lightness >= 40:
        tone = "Medium"
    else:
        tone = "Dark"

    # Lab a* and b* describe red-green and yellow-blue directions. Require a
    # 5-unit difference before assigning a directional undertone; near ties
    # are called Neutral. This is a photo-dependent heuristic, not a diagnosis.
    axis_difference = b_axis - a_axis
    if axis_difference >= 5:
        undertone = "Warm"
    elif axis_difference <= -5:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    return {
        "tone": tone,
        "undertone": undertone,
        "lab": (lightness, a_axis, b_axis),
    }


def assess_image_quality(
    image_rgb: np.ndarray,
    face_rect: tuple[int, int, int, int],
) -> list[str]:
    """Return non-blocking quality and framing notes for the detected face."""
    face_roi = crop_face(image_rgb, face_rect)
    image_height, image_width = image_rgb.shape[:2]
    x, y, face_width, face_height = face_rect
    gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    _, p10, p90, _ = np.percentile(gray, [5, 10, 90, 95])
    clipped_shadows = float(np.mean(gray <= 5))
    clipped_highlights = float(np.mean(gray >= 250))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    notes: list[str] = []

    if clipped_shadows >= 0.03 or p90 < 55:
        notes.append(
            "The face looks very dark or shadowed; even natural light may improve the estimate."
        )
    if clipped_highlights >= 0.05 or p10 > 235:
        notes.append(
            "Some facial highlights appear overexposed; soften direct light if possible."
        )
    if blur_score < 20:
        notes.append(
            "The face appears soft or blurry; a sharper image may improve the estimate."
        )
    if face_height / image_height < 0.10 or face_width / image_width < 0.10:
        notes.append(
            "The face occupies a small part of the image; move closer and keep it centered."
        )

    margin_x = max(2, round(image_width * 0.015))
    margin_y = max(2, round(image_height * 0.015))
    if (
        x <= margin_x
        or y <= margin_y
        or x + face_width >= image_width - margin_x
        or y + face_height >= image_height - margin_y
    ):
        notes.append(
            "The face is close to the image edge; leave a little space around the portrait."
        )

    return notes


def analyze_portrait(
    image_rgb: np.ndarray,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    """Run bounded face detection, facial color sampling, and heuristic estimates."""
    def report_progress(value: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    report_progress(8, "Finding a usable face…")
    face_rect, faces, only_small_faces = detect_face(image_rgb)
    if face_rect is None:
        if only_small_faces:
            raise PortraitAnalysisError(
                "A face was found, but it is too small to analyze reliably. "
                "Use a closer, higher-resolution portrait."
            )
        raise PortraitAnalysisError(
            "No usable face was detected. Use a clear, front-facing portrait with the face visible."
        )

    report_progress(42, "Sampling forehead and cheek color…")
    face_roi = crop_face(image_rgb, face_rect)
    representative_rgb, cluster_pixels, sampled_pixels = extract_skin_color(face_roi)

    report_progress(72, "Estimating tone and undertone…")
    classification = classify_skin_color(representative_rgb)
    color_hex = "#{:02X}{:02X}{:02X}".format(*representative_rgb)
    quality_warnings = assess_image_quality(image_rgb, face_rect)
    report_progress(100, "Analysis complete")

    return {
        "face_rect": face_rect,
        "face_count": len(faces),
        "multiple_faces": len(faces) > 1,
        "dominant_color": representative_rgb,
        "color_hex": color_hex,
        "tone": classification["tone"],
        "undertone": classification["undertone"],
        "lab": classification["lab"],
        "cluster_pixels": cluster_pixels,
        "sampled_pixels": sampled_pixels,
        "quality_warnings": quality_warnings,
    }
