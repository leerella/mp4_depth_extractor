from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "worker" / "vendor" / "video-depth-anything"
RUNTIME = ROOT / "worker" / "runtime"
MAX_BYTES = 500 * 1024 * 1024
MAX_DURATION = 60.0

app = FastAPI(title="DEPDY Worker", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def update_job(job_id: str, **values: object) -> None:
    with jobs_lock:
        jobs[job_id].update(values)


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
    }


def normalize_dimensions(width: int, height: int) -> tuple[int, int]:
    landscape = width >= height
    max_width, max_height = (1920, 1080) if landscape else (1080, 1920)
    long_edge, short_edge = max(width, height), min(width, height)
    if long_edge > 2048 or short_edge > 1100:
        raise ValueError("영상 해상도는 1080p를 넘을 수 없습니다.")

    scale = min(max_width / width, max_height / height, 1.0)
    normalized_width = max(2, round(width * scale / 2) * 2)
    normalized_height = max(2, round(height * scale / 2) * 2)
    return normalized_width, normalized_height


def create_green_screen_source(video_path: Path, matte_path: Path, target_path: Path) -> None:
    meta = probe_video(video_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"color=c=0x00ff00:s={meta['width']}x{meta['height']}:r={meta['fps']:g}:d={meta['duration']:g}",
            "-i", str(video_path), "-i", str(matte_path),
            "-filter_complex",
            "[1:v]format=rgba[subject];[2:v]format=gray[mask];"
            "[subject][mask]alphamerge[cutout];"
            "[0:v][cutout]overlay=shortest=1,format=yuv420p[green_screen]",
            "-map", "[green_screen]", "-c:v", "libx264", "-crf", "18",
            "-movflags", "+faststart", "-an", str(target_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def create_validation_sheet(
    video_path: Path,
    depth_path: Path,
    matte_path: Path | None,
    target_path: Path,
) -> None:
    meta = probe_video(video_path)
    frame_count = max(1, round(meta["duration"] * meta["fps"]))
    indices = sorted({0, frame_count // 3, (frame_count * 2) // 3, frame_count - 1})
    select = "+".join(f"eq(n,{index})" for index in indices)
    inputs = [video_path, depth_path]
    if matte_path is not None:
        inputs.append(matte_path)

    command = ["ffmpeg", "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    filters = [
        f"[{index}:v]select='{select}',scale=360:-2,tile=4x1[row{index}]"
        for index in range(len(inputs))
    ]
    rows = "".join(f"[row{index}]" for index in range(len(inputs)))
    filters.append(f"{rows}vstack=inputs={len(inputs)}[sheet]")
    command.extend(
        [
            "-filter_complex", ";".join(filters),
            "-map", "[sheet]", "-frames:v", "1", "-update", "1", str(target_path),
        ]
    )
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def process_sources(
    raw_depth: Path,
    normalized: Path,
    output_dir: Path,
    preset: str,
    invert: bool,
    contrast: float,
    brightness: float,
    highlights: float,
    shadows: float,
    matte: bool,
    preview_base: bool,
) -> None:
    command = [
        sys.executable, str(ROOT / "worker" / "process_sources.py"),
        "--depth", str(raw_depth), "--video", str(normalized),
        "--output-dir", str(output_dir), "--preset", preset,
        "--contrast", str(contrast), "--brightness", str(brightness),
        "--highlights", str(highlights), "--shadows", str(shadows),
    ]
    if invert:
        command.append("--invert")
    if matte:
        command.append("--matte")
    if preview_base:
        command.append("--preview-base")
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def validate_levels(
    preset: str,
    contrast: float,
    brightness: float,
    highlights: float,
    shadows: float,
) -> None:
    if preset not in {"natural", "subject", "contrast"}:
        raise HTTPException(422, "지원하지 않는 Depth 프리셋입니다.")
    if (
        not 0.5 <= contrast <= 2.0
        or not -50 <= brightness <= 50
        or not -100 <= highlights <= 100
        or not -100 <= shadows <= 100
    ):
        raise HTTPException(422, "Depth 레벨 값이 허용 범위를 벗어났습니다.")


def run_depth_job(
    job_id: str,
    input_path: Path,
    preset: str,
    invert: bool,
    contrast: float,
    brightness: float,
    highlights: float,
    shadows: float,
    matte: bool,
    alpha: bool,
) -> None:
    job_dir = input_path.parent
    output_dir = job_dir / "output"
    normalized = job_dir / "normalized.mp4"
    try:
        update_job(job_id, status="processing", progress=8, stage="영상 정보 확인")
        meta = probe_video(input_path)
        if meta["duration"] > MAX_DURATION:
            raise ValueError("영상 길이는 60초를 넘을 수 없습니다.")
        normalized_width, normalized_height = normalize_dimensions(meta["width"], meta["height"])

        target_fps = min(meta["fps"], 30.0)
        update_job(job_id, progress=15, stage="프레임 준비")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path), "-an",
                "-vf", f"fps={target_fps:g},scale={normalized_width}:{normalized_height}",
                "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", str(normalized),
            ],
            check=True,
            capture_output=True,
        )

        update_job(job_id, progress=25, stage="Depth 추론")
        command = [
            sys.executable, "run_streaming.py",
            "--input_video", str(normalized),
            "--output_dir", str(output_dir),
            "--encoder", "vits",
            "--input_size", "1008",
            "--max_res", "1920",
            "--grayscale",
        ]
        subprocess.run(
            command,
            cwd=VENDOR,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        generated = output_dir / "normalized_vis.mp4"
        if not generated.exists():
            raise RuntimeError("Depth 결과 파일이 생성되지 않았습니다.")

        raw_depth = output_dir / "raw_depth.mp4"
        shutil.move(generated, raw_depth)
        update_job(job_id, progress=72, stage="Depth 레벨 적용")
        if matte or alpha:
            update_job(job_id, progress=78, stage="Person Matte 생성")
        process_sources(
            raw_depth,
            normalized,
            output_dir,
            preset,
            invert,
            contrast,
            brightness,
            highlights,
            shadows,
            matte or alpha,
            True,
        )

        results: dict[str, object] = {
            "depth": f"/jobs/{job_id}/depth",
            "validation": f"/jobs/{job_id}/validation",
        }
        previews = {
            "depth": f"/jobs/{job_id}/preview/depth",
            "depthBase": f"/jobs/{job_id}/preview/depth-base",
        }
        if matte:
            results["matte"] = f"/jobs/{job_id}/matte"
            previews["matte"] = f"/jobs/{job_id}/preview/matte"
        if alpha:
            update_job(job_id, progress=92, stage="Green Screen 생성")
            create_green_screen_source(
                normalized,
                output_dir / "depdy_matte.mp4",
                output_dir / "depdy_green_screen.mp4",
            )
            results["alpha"] = f"/jobs/{job_id}/alpha"
            previews["alpha"] = f"/jobs/{job_id}/preview/alpha"
        create_validation_sheet(
            normalized,
            output_dir / "depdy_depth.mp4",
            output_dir / "depdy_matte.mp4" if matte else None,
            output_dir / "validation-sheet.jpg",
        )
        previews["validation"] = f"/jobs/{job_id}/preview/validation"
        results["previews"] = previews
        update_job(
            job_id,
            status="complete",
            progress=100,
            stage="완료",
            result_url=f"/jobs/{job_id}/depth",
            results=results,
            preset=preset,
            levels={
                "invert": invert,
                "contrast": contrast,
                "brightness": brightness,
                "highlights": highlights,
                "shadows": shadows,
            },
        )
    except subprocess.CalledProcessError as error:
        output = (error.stderr or error.stdout or "").strip().splitlines()
        detail = output[-1] if output else str(error)
        if "out of memory" in detail.lower():
            detail = "GPU 메모리가 부족합니다. 다른 GPU 작업을 종료한 뒤 다시 시도해주세요."
        update_job(job_id, status="failed", stage="오류", error=detail)
    except Exception as error:
        update_job(job_id, status="failed", stage="오류", error=str(error))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "video-depth-anything-small",
        "input_size": 1008,
        "profile": "quality",
    }


@app.post("/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    preset: str = Form("natural"),
    invert: bool = Form(False),
    contrast: float = Form(1.0),
    brightness: float = Form(0.0),
    highlights: float = Form(0.0),
    shadows: float = Form(0.0),
    matte: bool = Form(False),
    alpha: bool = Form(False),
) -> dict:
    validate_levels(preset, contrast, brightness, highlights, shadows)
    if video.content_type not in {"video/mp4", "video/quicktime", "application/octet-stream"}:
        raise HTTPException(415, "MP4 또는 MOV 영상만 올릴 수 있습니다.")

    job_id = uuid.uuid4().hex
    job_dir = RUNTIME / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / "input.mp4"
    size = 0
    with input_path.open("wb") as target:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "파일 크기는 500MB를 넘을 수 없습니다.")
            target.write(chunk)

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0, "stage": "대기 중"}
    background_tasks.add_task(
        run_depth_job,
        job_id,
        input_path,
        preset,
        invert,
        contrast,
        brightness,
        highlights,
        shadows,
        matte,
        alpha,
    )
    return jobs[job_id]


@app.post("/jobs/{job_id}/levels")
def update_levels(
    job_id: str,
    preset: str = Form("natural"),
    invert: bool = Form(False),
    contrast: float = Form(1.0),
    brightness: float = Form(0.0),
    highlights: float = Form(0.0),
    shadows: float = Form(0.0),
) -> dict:
    validate_levels(preset, contrast, brightness, highlights, shadows)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if job["status"] != "complete":
            raise HTTPException(409, "완료된 작업만 다시 조정할 수 있습니다.")

    job_dir = RUNTIME / job_id
    output_dir = job_dir / "output"
    raw_depth = output_dir / "raw_depth.mp4"
    normalized = job_dir / "normalized.mp4"
    if not raw_depth.exists() or not normalized.exists():
        raise HTTPException(404, "재조정에 필요한 원본 Depth를 찾을 수 없습니다.")

    try:
        preset_changed = job.get("preset") != preset
        process_sources(
            raw_depth,
            normalized,
            output_dir,
            preset,
            invert,
            contrast,
            brightness,
            highlights,
            shadows,
            False,
            preset_changed,
        )
        matte_path = output_dir / "depdy_matte.mp4"
        create_validation_sheet(
            normalized,
            output_dir / "depdy_depth.mp4",
            matte_path if "matte" in job["results"] else None,
            output_dir / "validation-sheet.jpg",
        )
    except subprocess.CalledProcessError as error:
        output = (error.stderr or error.stdout or "").strip().splitlines()
        raise HTTPException(500, output[-1] if output else str(error)) from error

    update_job(
        job_id,
        stage="Depth 레벨 업데이트 완료",
        preset=preset,
        levels={
            "invert": invert,
            "contrast": contrast,
            "brightness": brightness,
            "highlights": highlights,
            "shadows": shadows,
        },
    )
    return get_job(job_id)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        return dict(job)


@app.get("/jobs/{job_id}/depth")
def download_depth(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "depdy_depth.mp4"
    if not path.exists():
        raise HTTPException(404, "결과 파일을 찾을 수 없습니다.")
    return FileResponse(path, media_type="video/mp4", filename="depdy_depth.mp4")


@app.get("/jobs/{job_id}/matte")
def download_matte(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "depdy_matte.mp4"
    if not path.exists():
        raise HTTPException(404, "Person Matte 결과를 찾을 수 없습니다.")
    return FileResponse(path, media_type="video/mp4", filename="depdy_person_matte.mp4")


@app.get("/jobs/{job_id}/alpha")
def download_alpha(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "depdy_green_screen.mp4"
    if not path.exists():
        raise HTTPException(404, "Green Screen 결과를 찾을 수 없습니다.")
    return FileResponse(path, media_type="video/mp4", filename="depdy_green_screen.mp4")


@app.get("/jobs/{job_id}/validation")
def download_validation(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "validation-sheet.jpg"
    if not path.exists():
        raise HTTPException(404, "Validation Sheet를 찾을 수 없습니다.")
    return FileResponse(path, media_type="image/jpeg", filename="depdy_validation_sheet.jpg")


@app.get("/jobs/{job_id}/preview/depth")
def preview_depth(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "depdy_depth.mp4"
    if not path.exists():
        raise HTTPException(404, "Depth 미리보기를 찾을 수 없습니다.")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/jobs/{job_id}/preview/depth-base")
def preview_depth_base(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "depdy_depth_preview_base.mp4"
    if not path.exists():
        raise HTTPException(404, "실시간 Depth 미리보기를 찾을 수 없습니다.")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/jobs/{job_id}/preview/matte")
def preview_matte(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "depdy_matte.mp4"
    if not path.exists():
        raise HTTPException(404, "Person Matte 미리보기를 찾을 수 없습니다.")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/jobs/{job_id}/preview/alpha")
def preview_alpha(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "depdy_green_screen.mp4"
    if not path.exists():
        raise HTTPException(404, "Green Screen 미리보기를 찾을 수 없습니다.")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/jobs/{job_id}/preview/validation")
def preview_validation(job_id: str) -> FileResponse:
    path = RUNTIME / job_id / "output" / "validation-sheet.jpg"
    if not path.exists():
        raise HTTPException(404, "Validation Sheet 미리보기를 찾을 수 없습니다.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
