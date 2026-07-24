from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

# Ensures lineart_model/openpose_model resolve both when this file runs as a
# script (subprocess.run from app.py) and when imported as worker.process_sources
# (e.g. from tests), which don't put this directory on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lineart_model import detect_lineart, load_lineart_model
from openpose_model import detect_pose, load_pose_model


PRESET_CONTRAST = {"natural": 1.0, "subject": 1.18, "contrast": 1.45}
PRESET_BRIGHTNESS = {"natural": 0.0, "subject": 3.0, "contrast": 0.0}
MODEL_CACHE = Path(__file__).resolve().parent / "models" / "hub"
LINEART_CACHE = Path(__file__).resolve().parent / "models" / "lineart"
POSE_CACHE = Path(__file__).resolve().parent / "models" / "openpose"


def sequence_frame_dir(target: Path) -> Path:
    return target.with_name(f".{target.stem}_frames")


def write_frame_png(path: Path, frame: np.ndarray) -> None:
    # cv2.imwrite silently fails on Windows paths containing non-ASCII characters
    # (e.g. the project lives under a Korean username); encode + write bytes instead.
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"프레임을 PNG로 인코딩하지 못했습니다: {path}")
    path.write_bytes(encoded.tobytes())


def zip_sequence(frame_dir: Path, target_zip: Path) -> None:
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for frame_path in sorted(frame_dir.glob("*.png")):
            archive.write(frame_path, frame_path.name)
    shutil.rmtree(frame_dir)


def writer(path: Path, fps: float, size: tuple[int, int], color: bool = False) -> cv2.VideoWriter:
    result = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, color)
    if not result.isOpened():
        raise RuntimeError(f"출력 영상을 열 수 없습니다: {path}")
    return result


def encode_browser_mp4(source: Path, target: Path) -> None:
    encoded = target.with_name(f".{target.stem}.encoded.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(encoded),
            ],
            check=True,
        )
        encoded.replace(target)
    finally:
        encoded.unlink(missing_ok=True)
        source.unlink(missing_ok=True)


def adjust_depth(
    source: Path,
    target: Path,
    preset: str,
    invert: bool,
    contrast: float,
    brightness: float,
    highlights: float = 0.0,
    shadows: float = 0.0,
) -> None:
    working = target.with_name(f".{target.stem}.working.mp4")
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = writer(working, fps, (width, height))
    gain = PRESET_CONTRAST[preset] * contrast
    offset = PRESET_BRIGHTNESS[preset] + brightness

    frame_dir = sequence_frame_dir(target)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if preset == "subject":
            low, high = np.percentile(gray, (8, 92))
            if high > low:
                gray = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        adjusted = np.clip(gray.astype(np.float32) * gain + offset, 0, 255)
        normalized = adjusted / 255.0
        tone_delta = (
            (shadows / 100.0) * np.square(1.0 - normalized)
            + (highlights / 100.0) * np.square(normalized)
        ) * 0.35
        adjusted = np.round(np.clip(normalized + tone_delta, 0, 1) * 255).astype(np.uint8)
        if invert:
            adjusted = 255 - adjusted
        output.write(adjusted)
        write_frame_png(frame_dir / f"{index:05d}.png", adjusted)
        index += 1

    capture.release()
    output.release()
    encode_browser_mp4(working, target)
    zip_sequence(frame_dir, target.with_name(f"{target.stem}_sequence.zip"))


def create_preview_base(source: Path, target: Path, preset: str) -> None:
    working = target.with_name(f".{target.stem}.working.mp4")
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = writer(working, fps, (width, height))

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if preset == "subject":
            low, high = np.percentile(gray, (8, 92))
            if high > low:
                gray = np.clip(
                    (gray.astype(np.float32) - low) * 255.0 / (high - low),
                    0,
                    255,
                ).astype(np.uint8)
        output.write(gray)

    capture.release()
    output.release()
    encode_browser_mp4(working, target)


def create_matte(source: Path, target: Path) -> None:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(MODEL_CACHE))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(device).eval()
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    working = target.with_name(f".{target.stem}.working.mp4")
    output = writer(working, fps, (width, height))
    previous: np.ndarray | None = None

    with torch.inference_mode():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            scale = min(1.0, 720.0 / max(width, height))
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
            tensor = (tensor - torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]) / torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
            logits = model(tensor.unsqueeze(0))["out"][0]
            probability = torch.softmax(logits, dim=0)[15].cpu().numpy()
            mask = cv2.resize(probability, (width, height), interpolation=cv2.INTER_CUBIC)
            mask = np.clip((mask - 0.12) / 0.76, 0, 1)
            mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
            if previous is not None:
                mask = previous * 0.18 + mask * 0.82
            previous = mask
            output.write(np.round(mask * 255).astype(np.uint8))

    capture.release()
    output.release()
    encode_browser_mp4(working, target)


def create_frame_pass(source: Path, target: Path, load_model, detect, cache_dir: Path, color: bool = False) -> None:
    """Runs a per-frame detector (lineart/pose) over `source`, writing both a video and a PNG sequence zip."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cache_dir, device)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    working = target.with_name(f".{target.stem}.working.mp4")
    output = writer(working, fps, (width, height), color=color)

    frame_dir = sequence_frame_dir(target)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = detect(model, frame, device)
        output.write(result)
        write_frame_png(frame_dir / f"{index:05d}.png", result)
        index += 1

    capture.release()
    output.release()
    encode_browser_mp4(working, target)
    zip_sequence(frame_dir, target.with_name(f"{target.stem}_sequence.zip"))


def create_lineart(source: Path, target: Path) -> None:
    create_frame_pass(source, target, load_lineart_model, detect_lineart, LINEART_CACHE)


def create_pose(source: Path, target: Path) -> None:
    create_frame_pass(source, target, load_pose_model, detect_pose, POSE_CACHE, color=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESET_CONTRAST, default="natural")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--highlights", type=float, default=0.0)
    parser.add_argument("--shadows", type=float, default=0.0)
    parser.add_argument("--matte", action="store_true")
    parser.add_argument("--lineart", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--preview-base", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjust_depth(
        args.depth,
        args.output_dir / "depdy_depth.mp4",
        args.preset,
        args.invert,
        args.contrast,
        args.brightness,
        args.highlights,
        args.shadows,
    )
    if args.preview_base:
        create_preview_base(
            args.depth,
            args.output_dir / "depdy_depth_preview_base.mp4",
            args.preset,
        )
    if args.matte:
        create_matte(args.video, args.output_dir / "depdy_matte.mp4")
    if args.lineart:
        create_lineart(args.video, args.output_dir / "depdy_lineart.mp4")
    if args.pose:
        create_pose(args.video, args.output_dir / "depdy_pose.mp4")


if __name__ == "__main__":
    main()
