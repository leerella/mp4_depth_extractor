# DEPDY

로컬 GPU에서 MP4/MOV 영상의 Relative Depth, Person Matte, Green Screen, Line Art 소스와 Validation Sheet를 생성하는 웹 도구입니다.

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

필수 조건은 Windows, Python 3.11, Node.js, NVIDIA CUDA 환경, `ffmpeg`와 `ffprobe`(PATH에 등록)입니다.

```powershell
git clone https://github.com/leerella/mp4_depth_extractor.git
Set-Location mp4_depth_extractor
.\setup.ps1
```

`setup.ps1`이 서브모듈 초기화, Python 가상환경(`.venv`), `worker/requirements.txt` 설치, Video Depth Anything Small 가중치 다운로드, `web`의 `npm ci`까지 한 번에 처리합니다. Person Matte 모델과 Line Art 모델은 (용량이 작아) 최초 실행 시 자동으로 `worker/models`에 내려받아지며 Git에는 포함되지 않습니다.

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

1. 최대 60초, 500MB 이하의 MP4 또는 MOV 영상을 선택합니다.
2. Depth preset과 Person Matte/Green Screen/Line Art 출력을 선택합니다.
3. `Extract sources`를 누릅니다.
4. Depth 프리뷰에서 Levels와 Invert를 실시간으로 확인합니다.
5. Depth, Person Matte, Green Screen, Line Art, Validation Sheet를 각각 다운로드합니다. Depth와 Line Art는 PNG 시퀀스(ZIP)로도 받을 수 있습니다.

모든 입력과 결과는 `worker/runtime` 아래에서만 처리되며 Git에는 포함되지 않습니다.
