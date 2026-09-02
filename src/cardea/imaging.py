"""Reading DICOM clips and turning frames into API image parts."""

import base64
from io import BytesIO

import numpy as np
import pydicom
from PIL import Image

MAX_CANDIDATES = 50  # matches training: the keyframe selector saw up to 50 frames per clip


def read_dcm_3d_array(path):
    """Read a DICOM clip as a (frames, H, W) array, or None if it isn't one."""
    try:
        arr = pydicom.dcmread(path).pixel_array
    except Exception as e:  # noqa: BLE001 - invalid DICOM inputs may fail in several libraries
        print(f"[imaging] {path}: {e}")
        return None
    if arr.ndim != 3 or arr.shape[1] < 28 or arr.shape[2] < 28:
        return None
    return arr


def subsample_frames(video, max_frames=MAX_CANDIDATES):
    if video is None or len(video) == 0:
        return np.array([])
    if len(video) <= max_frames:
        return video
    return video[np.linspace(0, len(video) - 1, max_frames, dtype=int)]


def _to_uint8(frame):
    f = frame.astype(float)
    lo, hi = f.min(), f.max()
    if hi == lo:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) / (hi - lo) * 255).astype(np.uint8)


def frame_to_square_pil(frame, edge):
    return Image.fromarray(_to_uint8(frame)).resize((edge, edge), Image.Resampling.LANCZOS)


def _data_url(image, quality=95):
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_part(image):
    return {"type": "image_url", "image_url": {"url": _data_url(image)}}
