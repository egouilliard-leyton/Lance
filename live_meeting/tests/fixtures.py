"""Self-contained fixture synthesizers — NO binaries are committed to the repo.

Generates a tiny mp4 (via imageio+ffmpeg) and wav (via stdlib ``wave``) at test time.

Because mp4/H.264 (yuv420p) is lossy, exact RGB equality is unreliable. Tests should classify a
frame's *dominant* color against ``palette()`` using ``nearest_color_index`` rather than assert
exact pixel values. The palette colors are deliberately well separated so classification is robust.
"""

from __future__ import annotations

import wave
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np

_PALETTE: List[Tuple[int, int, int]] = [
    (220, 20, 20),    # red
    (20, 200, 20),    # green
    (20, 20, 220),    # blue
    (220, 200, 20),   # yellow
    (20, 200, 200),   # cyan
    (200, 20, 200),   # magenta
    (240, 240, 240),  # white
    (20, 20, 20),     # black
]


def palette() -> List[Tuple[int, int, int]]:
    return list(_PALETTE)


def frame_color(i: int) -> Tuple[int, int, int]:
    return _PALETTE[i % len(_PALETTE)]


def synth_mp4(path, n_frames: int = 8, fps: int = 1, size: Tuple[int, int] = (64, 64)) -> List[Tuple[int, int, int]]:
    """Write an mp4 of ``n_frames`` solid-color frames at ``fps``. Returns the intended colors."""
    h, w = size
    colors: List[Tuple[int, int, int]] = []
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1) as writer:
        for i in range(n_frames):
            c = frame_color(i)
            frame = np.empty((h, w, 3), dtype=np.uint8)
            frame[:, :] = c
            writer.append_data(frame)
            colors.append(c)
    return colors


def synth_wav(path, seconds: float = 8.0, sr: int = 16000, freq: float = 440.0) -> int:
    """Write a mono s16le sine wav. Returns the number of samples written."""
    n = int(sr * seconds)
    t = np.arange(n) / float(sr)
    data = (0.2 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return n


def dominant_color(image) -> Tuple[int, int, int]:
    flat = np.asarray(image).reshape(-1, np.asarray(image).shape[-1])
    med = np.median(flat, axis=0)
    return tuple(int(round(float(x))) for x in med[:3])


def nearest_color_index(image, pal: List[Tuple[int, int, int]] = None) -> int:
    pal = pal if pal is not None else _PALETTE
    dc = np.array(dominant_color(image)[:3], dtype=float)
    dists = [float(np.sum((np.array(c, dtype=float) - dc) ** 2)) for c in pal]
    return int(np.argmin(dists))
