# DEPDY

로컬 GPU에서 MP4/MOV 영상의 Relative Depth, Person Matte, Green Screen, Line Art, Pose Skeleton 소스와 Validation Sheet를 생성하는 웹 도구입니다.

## 실행

Windows PowerShell에서 프로젝트 루트로 이동한 뒤 실행합니다.

```powershell
.\start-depdy.ps1
```

스크립트는 반드시 이 프로젝트의 `.venv`를 사용하고, 프로덕션 웹을 빌드한 뒤 다음 서비스를 시작합니다.

- 웹: http://localhost:3000
- 워커 상태: http://127.0.0.1:8000/health

종료할 때는 다음 명령을 사용합니다.

```powershell
.\stop-depdy.ps1
```

PowerShell 실행 정책 때문에 차단되는 경우에만 다음처럼 실행할 수 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\start-depdy.ps1
```

## 최초 설치

필수 조건은 Windows, **Python 3.11**, Node.js, NVIDIA CUDA 환경, `ffmpeg`와 `ffprobe`(PATH에 등록)입니다.

**Python은 반드시 3.11이어야 합니다.** `worker/requirements.txt`에 고정된 `torch==2.1.1`, `xformers==0.0.23`는 Windows용 wheel을 cp311(3.11)까지만 배포하고 3.12 이상은 없어서, 3.12/3.13으로 설치하면 `pip install`이 실패합니다. 이미 다른 버전(3.12 등)이 설치돼 있어도 상관없이 3.11을 추가로 설치하면 되며, `py -3.11`로 버전을 골라 쓰기 때문에 서로 충돌하지 않습니다.

- 다운로드: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
- 설치 시 **"Add python.exe to PATH"** 체크
- 설치 후 새 PowerShell 창에서 `py -3.11 --version`으로 확인

```powershell
git clone https://github.com/leerella/mp4_depth_extractor.git
Set-Location mp4_depth_extractor
.\setup.ps1
```

`setup.ps1`이 서브모듈 초기화, Python 가상환경(`.venv`), `worker/requirements.txt` 설치, Video Depth Anything Small 가중치 다운로드, `web`의 `npm ci`까지 한 번에 처리합니다. Person Matte, Line Art, Pose Skeleton 모델은 최초 실행 시 자동으로 `worker/models`에 내려받아지며 Git에는 포함되지 않습니다.

**라이선스 참고**: Pose Skeleton은 CMU OpenPose 체크포인트(`lllyasviel/Annotators`의 `body_pose_model.pth`)를 사용합니다. 이 가중치는 **비상업적 용도로만** 사용 가능합니다 (CMU 라이선스). 상업적으로 사용하려면 별도 라이선스가 필요합니다. Depth/Person Matte/Line Art 모델에는 이 제약이 없습니다.

수동으로 설치하려면 아래 단계를 개별 실행해도 됩니다.

```powershell
git submodule update --init --recursive
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r worker\requirements.txt
Set-Location web
npm ci
Set-Location ..
```

Video Depth Anything Small 가중치는 아래 위치에 두면 됩니다 (다운로드 주소는 vendored 프로젝트의 `worker/vendor/video-depth-anything/README.md`에도 있습니다).

```text
worker\vendor\video-depth-anything\checkpoints\video_depth_anything_vits.pth
```

## 사용

1. 최대 60초, 500MB 이하의 MP4/MOV 영상 또는 JPG/PNG/WEBP 이미지를 선택합니다. 이미지는 내부적으로 짧은 영상으로 변환되어 동일한 파이프라인으로 처리됩니다.
2. Depth preset과 Person Matte/Green Screen/Line Art/Pose Skeleton 출력을 선택합니다. Pose Skeleton은 매우 느리므로(2080 Ti 기준 10초 영상 약 4분, 60초 영상 약 25분) 기본값이 꺼져 있습니다.
3. `Extract sources`를 누릅니다.
4. Depth 프리뷰에서 Levels와 Invert를 실시간으로 확인합니다.
5. Depth, Person Matte, Green Screen, Line Art, Pose Skeleton, Validation Sheet를 각각 다운로드합니다. Depth, Line Art, Pose Skeleton은 PNG 시퀀스(ZIP)로도 받을 수 있습니다.

모든 입력과 결과는 `worker/runtime` 아래에서만 처리되며 Git에는 포함되지 않습니다.
