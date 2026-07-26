# DEPDY

로컬 GPU에서 MP4/MOV 영상의 Relative Depth, Person Matte, Green Screen, Line Art, Pose Skeleton 소스와 Validation Sheet를 생성하는 웹 도구입니다.

## 🚀 빠른 시작 (처음 설치하는 경우)

아래 순서대로 하나씩 따라 하면 됩니다. **각 단계가 끝났는지 확인하고 다음으로 넘어가세요.**

### 1단계. 준비물 설치 (한 번만 하면 됨)

1. **NVIDIA 그래픽카드 드라이버** — GTX/RTX 그래픽카드가 있고 최신 드라이버만 설치돼 있으면 됩니다. CUDA Toolkit을 따로 설치할 필요는 없습니다 (필요한 부분이 자동으로 같이 설치됩니다).
2. **Python 3.11** (⚠️ 반드시 3.11, 3.12/3.13 아님)
   - 다운로드: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
   - 설치 화면에서 **"Add python.exe to PATH"** 체크박스를 꼭 체크
   - 이미 다른 버전(3.12 등)이 깔려 있어도 상관없이 3.11을 추가로 설치하면 됩니다
3. **Node.js** (LTS 버전) — 다운로드: https://nodejs.org/en/download
4. **Git** — 다운로드: https://git-scm.com/download/win
5. **ffmpeg** — PowerShell을 열고 아래 명령을 실행합니다.
   ```powershell
   winget install Gyan.FFmpeg
   ```

설치가 다 끝나면 **PowerShell 창을 전부 닫고 새로 하나 엽니다** (안 그러면 방금 설치한 프로그램들을 못 찾습니다). 새 창에서 아래 명령들로 잘 설치됐는지 확인하세요.

```powershell
py -3.11 --version
node --version
git --version
ffmpeg -version
```

네 개 다 에러 없이 버전이 나오면 준비 완료입니다.

### 2단계. 다운로드 및 설치 (한 번만 하면 됨)

PowerShell에서 프로젝트를 받을 폴더로 이동한 뒤 (예: `cd Desktop`), 아래 3줄을 **한 줄씩 직접 타이핑하지 말고 그대로 복사해서 붙여넣으세요** (타이핑하다 오타가 나면 엉뚱한 에러가 납니다).

```powershell
git clone https://github.com/leerella/mp4_depth_extractor.git
Set-Location mp4_depth_extractor
.\setup.ps1
```

`setup.ps1`이 필요한 걸 전부 알아서 내려받고 설치합니다 (Depth 모델 가중치 다운로드 때문에 몇 분 걸릴 수 있습니다). 마지막에 `5/5 Done`이 뜨면 성공입니다.

### 3단계. 실행 (쓸 때마다 이것만 하면 됨)

```powershell
.\start-depdy.ps1
```

자동으로 브라우저가 열립니다. 안 열리면 직접 **http://localhost:3002** 을 주소창에 입력하세요. 다 쓰고 끝낼 때는 `.\stop-depdy.ps1`을 실행합니다.

### 문제가 생겼다면

| 증상 | 원인 / 해결 |
|---|---|
| `No suitable Python runtime found` | Python 3.11이 설치 안 된 것입니다. 1단계 2번부터 다시 하세요. |
| `'...python.exe' 용어가 cmdlet, 함수... 인식되지 않습니다` | 보통 위 원인(3.11 미설치)의 후속 에러입니다. 마찬가지로 3.11부터 다시 설치하세요. |
| `It : 'test' 매개 변수...` 같은 알 수 없는 에러 | 명령어를 손으로 한 줄씩 타이핑하다 오타(`git`이 `it`으로 입력되는 등)가 난 경우입니다. `.\setup.ps1`, `.\start-depdy.ps1`처럼 **스크립트 파일 자체를 실행**하세요. 내부 명령어를 직접 타이핑하지 마세요. |
| 브라우저에 "사이트에 연결할 수 없음" | 서버가 꺼져 있는 것입니다. `.\start-depdy.ps1`을 다시 실행하고, 뜬 창을 닫지 말고 그대로 두세요. |
| `pip install` 관련 실패 | `py -3.11 --version`을 실행해서 3.11이 맞는지 확인하세요. |
| PowerShell 실행 정책 에러 | 아래처럼 실행하세요. |

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

---

## 실행 (참고)

Windows PowerShell에서 프로젝트 루트로 이동한 뒤 실행합니다.

```powershell
.\start-depdy.ps1
```

스크립트는 반드시 이 프로젝트의 `.venv`를 사용하고, 프로덕션 웹을 빌드한 뒤 다음 서비스를 시작합니다.

- 웹: http://localhost:3002
- 워커 상태: http://127.0.0.1:8000/health

종료할 때는 다음 명령을 사용합니다.

```powershell
.\stop-depdy.ps1
```

## 수동 설치 (참고, 보통은 필요 없음)

`setup.ps1` 대신 아래 단계를 개별 실행할 수도 있습니다.

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

Person Matte, Line Art, Pose Skeleton 모델은 최초 실행 시 자동으로 `worker/models`에 내려받아지며 Git에는 포함되지 않습니다.

**라이선스 참고**: Pose Skeleton은 CMU OpenPose 체크포인트(`lllyasviel/Annotators`의 `body_pose_model.pth`)를 사용합니다. 이 가중치는 **비상업적 용도로만** 사용 가능합니다 (CMU 라이선스). 상업적으로 사용하려면 별도 라이선스가 필요합니다. Depth/Person Matte/Line Art 모델에는 이 제약이 없습니다.

## 사용

1. 최대 60초, 500MB 이하의 MP4/MOV 영상 또는 JPG/PNG/WEBP 이미지를 선택합니다. 이미지는 내부적으로 짧은 영상으로 변환되어 동일한 파이프라인으로 처리됩니다.
2. Depth preset과 Person Matte/Green Screen/Line Art/Pose Skeleton 출력을 선택합니다. Pose Skeleton은 매우 느리므로(2080 Ti 기준 10초 영상 약 4분, 60초 영상 약 25분) 기본값이 꺼져 있습니다.
3. `Extract sources`를 누릅니다.
4. Depth 프리뷰에서 Levels와 Invert를 실시간으로 확인합니다.
5. Depth, Person Matte, Green Screen, Line Art, Pose Skeleton, Validation Sheet를 각각 다운로드합니다. Depth, Line Art, Pose Skeleton은 PNG 시퀀스(ZIP)로도 받을 수 있습니다.

모든 입력과 결과는 `worker/runtime` 아래에서만 처리되며 Git에는 포함되지 않습니다.
