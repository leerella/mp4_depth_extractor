from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from worker.app import create_green_screen_source, create_validation_sheet
from worker.process_sources import adjust_depth, create_matte, create_preview_base


ROOT = Path(__file__).resolve().parent.parent
REAL_VIDEO = ROOT / "ref1.mp4"


def read_gray_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    if not frames:
        raise AssertionError(f"No decodable frames in {path}")
    return np.stack(frames)


def read_color_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise AssertionError(f"No decodable frames in {path}")
    return np.stack(frames)


def make_real_clip(target: Path, frame_limit: int = 6) -> None:
    capture = cv2.VideoCapture(str(REAL_VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for _ in range(frame_limit):
        ok, frame = capture.read()
        if not ok:
            break
        output.write(frame)
    capture.release()
    output.release()


def probe_stream(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,codec_tag_string,pix_fmt,nb_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


class FeatureIntegrationTests(unittest.TestCase):
    def test_depth_levels_change_real_frames(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "worker") as temp_dir:
            directory = Path(temp_dir)
            clip = directory / "clip.mp4"
            make_real_clip(clip)
            natural = directory / "natural.mp4"
            dark = directory / "dark.mp4"
            bright = directory / "bright.mp4"
            contrast = directory / "contrast.mp4"
            highlights = directory / "highlights.mp4"
            shadows = directory / "shadows.mp4"
            inverted = directory / "inverted.mp4"
            preview_base = directory / "preview-base.mp4"

            adjust_depth(clip, natural, "natural", False, 1.0, 0.0)
            adjust_depth(clip, dark, "natural", False, 1.0, -50.0)
            adjust_depth(clip, bright, "natural", False, 1.0, 50.0)
            adjust_depth(clip, contrast, "natural", False, 1.5, 0.0)
            adjust_depth(clip, highlights, "natural", False, 1.0, 0.0, 70.0, 0.0)
            adjust_depth(clip, shadows, "natural", False, 1.0, 0.0, 0.0, 70.0)
            adjust_depth(clip, inverted, "natural", True, 1.0, 0.0)
            create_preview_base(clip, preview_base, "subject")

            base_frames = read_gray_video(natural)
            dark_frames = read_gray_video(dark)
            bright_frames = read_gray_video(bright)
            contrast_frames = read_gray_video(contrast)
            highlight_frames = read_gray_video(highlights)
            shadow_frames = read_gray_video(shadows)
            inverted_frames = read_gray_video(inverted)
            self.assertEqual(len(base_frames), 6)
            self.assertLess(float(dark_frames.mean()), float(base_frames.mean()) - 35)
            self.assertGreater(float(bright_frames.mean()), float(base_frames.mean()) + 35)
            self.assertGreater(
                float(np.abs(contrast_frames.astype(float) - base_frames).mean()), 20
            )
            low_pixels = base_frames < 85
            high_pixels = base_frames > 170
            highlight_delta = np.abs(highlight_frames.astype(float) - base_frames)
            shadow_delta = np.abs(shadow_frames.astype(float) - base_frames)
            self.assertGreater(
                float(highlight_delta[high_pixels].mean()),
                float(highlight_delta[low_pixels].mean()) * 2,
            )
            self.assertGreater(
                float(shadow_delta[low_pixels].mean()),
                float(shadow_delta[high_pixels].mean()) * 2,
            )
            self.assertGreater(
                float(np.abs(inverted_frames.astype(float) - base_frames).mean()), 50
            )
            for output in (natural, dark, bright, contrast, highlights, shadows, inverted):
                stream = probe_stream(output)
                self.assertEqual(stream["codec_name"], "h264")
                self.assertEqual(stream["codec_tag_string"], "avc1")
                self.assertEqual(stream["pix_fmt"], "yuv420p")
            preview_stream = probe_stream(preview_base)
            self.assertEqual(preview_stream["codec_name"], "h264")
            self.assertEqual(int(preview_stream["nb_frames"]), 6)

    def test_person_matte_and_green_screen_on_real_frames(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "worker") as temp_dir:
            directory = Path(temp_dir)
            clip = directory / "clip.mp4"
            matte = directory / "matte.mp4"
            green_screen = directory / "green-screen.mp4"
            validation = directory / "validation-sheet.jpg"
            make_real_clip(clip)

            create_matte(clip, matte)
            create_green_screen_source(clip, matte, green_screen)
            create_validation_sheet(clip, clip, matte, validation)
            matte_frames = read_gray_video(matte)
            green_frames = read_color_video(green_screen)
            foreground = float((matte_frames >= 128).mean())
            self.assertEqual(len(matte_frames), 6)
            self.assertEqual(len(green_frames), 6)
            self.assertGreater(foreground, 0.05)
            self.assertLess(foreground, 0.6)
            self.assertGreater(int(matte_frames.max()), 245)
            matte_stream = probe_stream(matte)
            self.assertEqual(matte_stream["codec_name"], "h264")
            self.assertEqual(matte_stream["codec_tag_string"], "avc1")
            self.assertEqual(matte_stream["pix_fmt"], "yuv420p")
            sheet = cv2.imdecode(np.fromfile(validation, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(sheet)
            self.assertEqual(sheet.shape[1], 1440)

            stream = probe_stream(green_screen)
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(stream["codec_tag_string"], "avc1")
            self.assertEqual(stream["pix_fmt"], "yuv420p")
            self.assertEqual(int(stream["nb_frames"]), 6)

            background = green_frames[matte_frames < 16]
            self.assertGreater(float(background[:, 1].mean()), 240)
            self.assertLess(float(background[:, 0].mean()), 16)
            self.assertLess(float(background[:, 2].mean()), 16)


if __name__ == "__main__":
    unittest.main()
