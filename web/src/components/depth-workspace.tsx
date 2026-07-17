"use client";

import {
  ArrowCounterClockwise,
  ArrowRight,
  Check,
  DownloadSimple,
  FileVideo,
  SlidersHorizontal,
  UploadSimple,
  X,
} from "@phosphor-icons/react";
import Image from "next/image";
import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

const MAX_BYTES = 500 * 1024 * 1024;
const WORKER_URL = "http://127.0.0.1:8000";

type Preset = "natural" | "subject" | "contrast";
type ResultKind = "depth" | "matte" | "alpha" | "validation";
type LevelSettings = {
  invert: boolean;
  contrast: number;
  brightness: number;
  highlights: number;
  shadows: number;
};
type JobResults = {
  depth: string;
  matte?: string;
  alpha?: string;
  validation: string;
  previews: {
    depth: string;
    depthBase?: string;
    matte?: string;
    alpha?: string;
    validation: string;
  };
};
type Job = {
  id: string;
  status: "queued" | "processing" | "complete" | "failed";
  progress: number;
  stage: string;
  result_url?: string;
  results?: JobResults;
  preset?: Preset;
  levels?: LevelSettings;
  error?: string;
};

const DEFAULT_LEVELS: LevelSettings = {
  invert: false,
  contrast: 1,
  brightness: 0,
  highlights: 0,
  shadows: 0,
};

function getLevelSignature(preset: Preset, levels: LevelSettings) {
  return [
    preset,
    levels.invert,
    levels.contrast,
    levels.brightness,
    levels.highlights,
    levels.shadows,
  ].join(":");
}

const PRESET_GAIN: Record<Preset, number> = {
  natural: 1,
  subject: 1.18,
  contrast: 1.45,
};
const PRESET_OFFSET: Record<Preset, number> = {
  natural: 0,
  subject: 3,
  contrast: 0,
};

function createDepthToneTable(preset: Preset, levels: LevelSettings) {
  return Array.from({ length: 256 }, (_, index) => {
    let value = Math.min(1, Math.max(0, (
      index * PRESET_GAIN[preset] * levels.contrast
      + PRESET_OFFSET[preset]
      + levels.brightness
    ) / 255));
    value += (
      (levels.shadows / 100) * (1 - value) ** 2
      + (levels.highlights / 100) * value ** 2
    ) * 0.35;
    value = Math.round(Math.min(1, Math.max(0, value)) * 255) / 255;
    if (levels.invert) value = 1 - value;
    return value.toFixed(4);
  }).join(" ");
}

const PRESET_DETAILS: Record<Preset, { label: string; image: string; description: string }> = {
  natural: {
    label: "Natural",
    image: "/presets/natural.webp",
    description: "원본의 깊이 그라데이션을 가장 자연스럽게 유지합니다.",
  },
  subject: {
    label: "Subject",
    image: "/presets/subject.webp",
    description: "인물 주변의 깊이 범위를 확장해 피사체를 또렷하게 분리합니다.",
  },
  contrast: {
    label: "Contrast",
    image: "/presets/contrast.webp",
    description: "명암 차이를 강하게 만들어 합성과 변위 작업에 적합합니다.",
  },
};

const RESULT_LABELS: Record<ResultKind, string> = {
  depth: "Depth",
  matte: "Person Matte",
  alpha: "Green Screen",
  validation: "Validation Sheet",
};

export function DepthWorkspace() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [preset, setPreset] = useState<Preset>("natural");
  const [matte, setMatte] = useState(true);
  const [alpha, setAlpha] = useState(true);
  const [invert, setInvert] = useState(false);
  const [contrastLevel, setContrastLevel] = useState(1);
  const [brightness, setBrightness] = useState(0);
  const [highlights, setHighlights] = useState(0);
  const [shadows, setShadows] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [activeResult, setActiveResult] = useState<ResultKind>("depth");
  const [previewVersion, setPreviewVersion] = useState(0);
  const [downloadVersion, setDownloadVersion] = useState(0);
  const [previewError, setPreviewError] = useState<{ url: string; message: string } | null>(null);
  const [applyingLevels, setApplyingLevels] = useState(false);
  const [lastAppliedSignature, setLastAppliedSignature] = useState<string | null>(null);
  const [autoApplyBlockedSignature, setAutoApplyBlockedSignature] = useState<string | null>(null);

  const currentLevelSignature = getLevelSignature(preset, {
    invert,
    contrast: contrastLevel,
    brightness,
    highlights,
    shadows,
  });
  const depthToneTable = useMemo(() => createDepthToneTable(preset, {
    invert,
    contrast: contrastLevel,
    brightness,
    highlights,
    shadows,
  }), [brightness, contrastLevel, highlights, invert, preset, shadows]);
  const levelsAreDefault =
    invert === DEFAULT_LEVELS.invert &&
    contrastLevel === DEFAULT_LEVELS.contrast &&
    brightness === DEFAULT_LEVELS.brightness &&
    highlights === DEFAULT_LEVELS.highlights &&
    shadows === DEFAULT_LEVELS.shadows;
  const completedJobId = job?.status === "complete" ? job.id : null;

  useEffect(() => {
    if (!preview) return;
    return () => URL.revokeObjectURL(preview);
  }, [preview]);

  function acceptFile(candidate?: File) {
    setError(null);
    if (!candidate) return;
    if (!candidate.type.startsWith("video/")) {
      setError("MP4 또는 MOV 영상만 올릴 수 있습니다.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError("파일 크기는 500MB를 넘을 수 없습니다.");
      return;
    }
    setFile(candidate);
    setPreview(URL.createObjectURL(candidate));
  }

  function onInput(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  async function startExtraction() {
    if (!file || job?.status === "processing" || job?.status === "queued") return;
    setError(null);
    const submittedLevelSignature = currentLevelSignature;
    setJob({ id: "", status: "queued", progress: 0, stage: "업로드 중" });
    try {
      const body = new FormData();
      body.append("video", file);
      body.append("preset", preset);
      body.append("invert", String(invert));
      body.append("contrast", String(contrastLevel));
      body.append("brightness", String(brightness));
      body.append("highlights", String(highlights));
      body.append("shadows", String(shadows));
      body.append("matte", String(matte));
      body.append("alpha", String(alpha));
      const response = await fetch(`${WORKER_URL}/jobs`, { method: "POST", body });
      if (!response.ok) throw new Error(await response.text());
      const created = (await response.json()) as Job;
      setJob(created);

      const timer = window.setInterval(async () => {
        try {
          const statusResponse = await fetch(`${WORKER_URL}/jobs/${created.id}`);
          if (!statusResponse.ok) throw new Error("작업 상태를 확인할 수 없습니다.");
          const current = (await statusResponse.json()) as Job;
          setJob(current);
          if (current.status === "complete" || current.status === "failed") {
            window.clearInterval(timer);
            if (current.status === "complete") {
              setActiveResult("depth");
              setPreviewVersion((version) => version + 1);
              setDownloadVersion((version) => version + 1);
              setLastAppliedSignature(submittedLevelSignature);
              setAutoApplyBlockedSignature(null);
            }
          }
        } catch (pollError) {
          window.clearInterval(timer);
          setError(pollError instanceof Error ? pollError.message : "처리 상태 확인에 실패했습니다.");
        }
      }, 1200);
    } catch (requestError) {
      setJob(null);
      setError(requestError instanceof Error ? requestError.message : "업로드에 실패했습니다.");
    }
  }

  const applyDepthLevels = useCallback(async (
    expectedSignature = currentLevelSignature,
    autoApply = false,
  ) => {
    if (!completedJobId || applyingLevels) return;
    setApplyingLevels(true);
    setError(null);
    const presetChanged = preset !== job?.preset;
    try {
      const body = new FormData();
      body.append("preset", preset);
      body.append("invert", String(invert));
      body.append("contrast", String(contrastLevel));
      body.append("brightness", String(brightness));
      body.append("highlights", String(highlights));
      body.append("shadows", String(shadows));
      const response = await fetch(`${WORKER_URL}/jobs/${completedJobId}/levels`, {
        method: "POST",
        body,
      });
      if (!response.ok) throw new Error(await response.text());
      setJob((await response.json()) as Job);
      setActiveResult("depth");
      if (presetChanged) setPreviewVersion((version) => version + 1);
      setDownloadVersion((version) => version + 1);
      setLastAppliedSignature(expectedSignature);
      setAutoApplyBlockedSignature(null);
    } catch (levelError) {
      if (autoApply) setAutoApplyBlockedSignature(expectedSignature);
      setError(levelError instanceof Error ? levelError.message : "Depth 레벨 적용에 실패했습니다.");
    } finally {
      setApplyingLevels(false);
    }
  }, [
    applyingLevels,
    brightness,
    completedJobId,
    contrastLevel,
    currentLevelSignature,
    highlights,
    invert,
    job?.preset,
    preset,
    shadows,
  ]);

  useEffect(() => {
    if (
      !completedJobId ||
      applyingLevels ||
      currentLevelSignature === lastAppliedSignature ||
      currentLevelSignature === autoApplyBlockedSignature
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      void applyDepthLevels(currentLevelSignature, true);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [
    applyDepthLevels,
    applyingLevels,
    autoApplyBlockedSignature,
    completedJobId,
    currentLevelSignature,
    lastAppliedSignature,
  ]);

  function resetLevels() {
    setInvert(DEFAULT_LEVELS.invert);
    setContrastLevel(DEFAULT_LEVELS.contrast);
    setBrightness(DEFAULT_LEVELS.brightness);
    setHighlights(DEFAULT_LEVELS.highlights);
    setShadows(DEFAULT_LEVELS.shadows);
    setAutoApplyBlockedSignature(null);
    if (completedJobId) setActiveResult("depth");
  }

  const resultTabs = job?.status === "complete" && job.results
    ? ([
        {
          kind: "depth" as const,
          path: job.results.previews.depthBase ?? job.results.previews.depth,
        },
        job.results.matte && job.results.previews.matte
          ? { kind: "matte" as const, path: job.results.previews.matte }
          : null,
        job.results.alpha && job.results.previews.alpha
          ? { kind: "alpha" as const, path: job.results.previews.alpha }
          : null,
        { kind: "validation" as const, path: job.results.previews.validation },
      ].filter(Boolean) as { kind: ResultKind; path: string }[])
    : [];
  const activePreviewPath = resultTabs.find(({ kind }) => kind === activeResult)?.path;
  const activePreviewUrl = activePreviewPath
    ? `${WORKER_URL}${activePreviewPath}?v=${previewVersion}`
    : null;

  if (!file) {
    return (
      <div
        className={`group flex min-h-[620px] cursor-pointer flex-col justify-between rounded-[14px] border bg-[#101318] p-5 text-white transition-colors md:p-7 ${dragging ? "border-[#f5f5f3]" : "border-[#101318]"}`}
        onClick={() => inputRef.current?.click()}
        onDragEnter={() => setDragging(true)}
        onDragLeave={() => setDragging(false)}
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
      >
        <input ref={inputRef} type="file" accept="video/mp4,video/quicktime" onChange={onInput} className="sr-only" />
        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.12em] text-[#92969d]">
          <span>New extraction</span>
          <UploadSimple size={18} weight="regular" />
        </div>
        <div className="max-w-lg">
          <p className="text-3xl font-medium leading-tight tracking-[-0.045em] md:text-5xl">
            영상을 여기에 놓아주세요.
          </p>
          <p className="mt-5 max-w-sm text-sm leading-6 text-[#9da1a8]">
            클릭해서 파일을 선택하거나 드래그하세요. MP4, MOV 파일을 지원합니다.
          </p>
          {error && <p className="mt-4 text-sm text-[#ff8c7f]">{error}</p>}
        </div>
        <div className="flex items-center justify-between border-t border-white/20 pt-5 text-xs text-[#a5a8ad]">
          <span>최대 500MB</span>
          <span className="flex items-center gap-2 text-white">
            Select video <ArrowRight size={15} />
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-[14px] border border-[#c8c9c5] bg-white">
      <div className="relative aspect-video overflow-hidden bg-[#111317]">
        <svg className="absolute size-0" aria-hidden="true">
          <defs>
            <filter id="depth-levels-preview-filter" colorInterpolationFilters="sRGB">
              <feComponentTransfer>
                <feFuncR type="table" tableValues={depthToneTable} />
                <feFuncG type="table" tableValues={depthToneTable} />
                <feFuncB type="table" tableValues={depthToneTable} />
                <feFuncA type="identity" />
              </feComponentTransfer>
            </filter>
          </defs>
        </svg>
        {activePreviewUrl ? (
          activeResult === "validation" ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={activePreviewUrl}
              alt="Depth, Person Matte와 원본 프레임을 비교하는 Validation Sheet"
              className="h-full w-full object-contain"
            />
          ) : (
            <video
              key={activePreviewUrl}
              className="h-full w-full object-contain"
              style={activeResult === "depth" ? { filter: "url(#depth-levels-preview-filter)" } : undefined}
              controls
              autoPlay
              loop
              muted
              playsInline
              preload="auto"
              onCanPlay={() => setPreviewError(null)}
              onError={() => setPreviewError({
                url: activePreviewUrl,
                message: "이 결과 영상을 브라우저에서 재생할 수 없습니다. 다시 추출해 주세요.",
              })}
            >
              <source src={activePreviewUrl} type="video/mp4" />
              브라우저가 MP4 미리보기를 지원하지 않습니다.
            </video>
          )
        ) : preview && <video src={preview} className="h-full w-full object-contain" controls playsInline />}
        {previewError?.url === activePreviewUrl && (
          <div className="absolute inset-x-4 bottom-4 rounded-[9px] bg-[#b33c32]/95 px-4 py-3 text-xs text-white">
            {previewError.message}
          </div>
        )}
        {activeResult === "alpha" && activePreviewUrl && (
          <span className="absolute bottom-4 left-4 rounded-full bg-black/65 px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.08em] text-white backdrop-blur">
            Green screen source
          </span>
        )}
        {activeResult === "depth" && activePreviewUrl && (
          <span className="absolute bottom-4 left-4 rounded-full bg-black/65 px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.08em] text-white backdrop-blur">
            Live levels preview
          </span>
        )}
        <button
          onClick={() => {
            setFile(null);
            setPreview(null);
            setJob(null);
            setError(null);
            setActiveResult("depth");
            setLastAppliedSignature(null);
            setAutoApplyBlockedSignature(null);
          }}
          className="absolute right-4 top-4 grid size-9 place-items-center rounded-full bg-black/60 text-white backdrop-blur transition-transform active:scale-95"
          aria-label="영상 제거"
        >
          <X size={15} />
        </button>
      </div>

      {resultTabs.length > 0 && (
        <div className="flex gap-1 overflow-x-auto border-b border-[#d9dad6] bg-[#f4f4f1] p-2">
          {resultTabs.map(({ kind }) => (
            <button
              key={kind}
              type="button"
              onClick={() => setActiveResult(kind)}
              className={`shrink-0 rounded-[8px] px-3 py-2 text-[11px] font-medium transition-colors ${
                activeResult === kind
                  ? "bg-[#101318] text-white"
                  : "text-[#666a70] hover:bg-white hover:text-[#101318]"
              }`}
            >
              {RESULT_LABELS[kind]}
            </button>
          ))}
        </div>
      )}

      <div className="p-5 md:p-7">
        <div className="flex items-start justify-between gap-5 border-b border-[#d9dad6] pb-5">
          <div className="flex min-w-0 gap-3">
            <FileVideo className="mt-0.5 shrink-0" size={19} />
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{file.name}</p>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.08em] text-[#85888d]">
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </p>
            </div>
          </div>
          <SlidersHorizontal size={19} />
        </div>

        <fieldset className="mt-6">
          <legend className="mb-3 font-mono text-[10px] uppercase tracking-[0.1em] text-[#74777c]">Depth preset</legend>
          <div className="grid grid-cols-3 gap-2">
            {(["natural", "subject", "contrast"] as const).map((value) => {
              const detail = PRESET_DETAILS[value];
              return (
                <div key={value} className="group/preset relative">
                  <button
                    type="button"
                    onClick={() => {
                      setPreset(value);
                      setAutoApplyBlockedSignature(null);
                      if (completedJobId) setActiveResult("depth");
                    }}
                    aria-describedby={`preset-${value}-tip`}
                    className={`h-10 w-full rounded-[9px] border text-xs transition-colors active:scale-[0.98] ${preset === value ? "border-[#101318] bg-[#101318] text-white" : "border-[#d1d2ce] bg-white text-[#585c62] hover:border-[#8b8e92]"}`}
                  >
                    {detail.label}
                  </button>
                  <div
                    id={`preset-${value}-tip`}
                    role="tooltip"
                    className={`pointer-events-none invisible absolute bottom-[calc(100%+10px)] z-30 w-64 translate-y-1 overflow-hidden rounded-[10px] border border-white/15 bg-[#101318] opacity-0 shadow-2xl transition-all duration-150 group-hover/preset:visible group-hover/preset:translate-y-0 group-hover/preset:opacity-100 group-focus-within/preset:visible group-focus-within/preset:translate-y-0 group-focus-within/preset:opacity-100 ${
                      value === "natural" ? "left-0" : value === "contrast" ? "right-0" : "left-1/2 -translate-x-1/2"
                    }`}
                  >
                    <div className="relative aspect-[640/283] overflow-hidden">
                      <Image
                        src={detail.image}
                        alt={`${detail.label} Depth 프리셋 예시`}
                        fill
                        sizes="256px"
                        className="object-cover"
                      />
                      <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-1 font-mono text-[8px] uppercase tracking-[0.08em] text-white">
                        Source / Depth
                      </span>
                    </div>
                    <div className="p-3 text-left">
                      <p className="text-xs font-medium text-white">{detail.label}</p>
                      <p className="mt-1 text-[10px] leading-4 text-[#aeb2b9]">{detail.description}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </fieldset>

        <fieldset className="mt-6 border-t border-[#d9dad6] pt-5">
          <legend className="font-mono text-[10px] uppercase tracking-[0.1em] text-[#74777c]">Levels</legend>
          <div className="mt-2 flex justify-end">
            <button
              type="button"
              onClick={resetLevels}
              disabled={levelsAreDefault}
              className="flex items-center gap-1.5 rounded-[7px] px-2 py-1.5 text-[10px] font-medium text-[#676b72] transition-colors hover:bg-[#f1f1ed] hover:text-[#101318] disabled:cursor-default disabled:opacity-35"
            >
              <ArrowCounterClockwise size={13} />
              Reset to default
            </button>
          </div>
          <div className="mt-4 flex items-center justify-between">
            <span className="text-sm font-medium">Invert Depth</span>
            <button
              type="button"
              role="switch"
              aria-checked={invert}
              onClick={() => {
                setInvert(!invert);
                setAutoApplyBlockedSignature(null);
                if (completedJobId) setActiveResult("depth");
              }}
              className={`relative h-6 w-11 rounded-full transition-colors ${invert ? "bg-[#101318]" : "bg-[#c9ccd1]"}`}
            >
              <span className={`absolute top-1 size-4 rounded-full bg-white transition-transform ${invert ? "translate-x-6" : "translate-x-1"}`} />
            </button>
          </div>
          <LevelSlider label="Contrast" value={contrastLevel} min={0.5} max={2} step={0.05} display={`${contrastLevel.toFixed(2)}×`} onChange={(value) => { setContrastLevel(value); setAutoApplyBlockedSignature(null); if (completedJobId) setActiveResult("depth"); }} />
          <LevelSlider label="Brightness" value={brightness} min={-50} max={50} step={1} display={`${brightness > 0 ? "+" : ""}${brightness}`} onChange={(value) => { setBrightness(value); setAutoApplyBlockedSignature(null); if (completedJobId) setActiveResult("depth"); }} />
          <LevelSlider label="Highlights" value={highlights} min={-100} max={100} step={1} display={`${highlights > 0 ? "+" : ""}${highlights}`} onChange={(value) => { setHighlights(value); setAutoApplyBlockedSignature(null); if (completedJobId) setActiveResult("depth"); }} />
          <LevelSlider label="Shadows" value={shadows} min={-100} max={100} step={1} display={`${shadows > 0 ? "+" : ""}${shadows}`} onChange={(value) => { setShadows(value); setAutoApplyBlockedSignature(null); if (completedJobId) setActiveResult("depth"); }} />
          {job?.status === "complete" && (
            <div className="mt-5">
              <button
                type="button"
                onClick={() => {
                  setAutoApplyBlockedSignature(null);
                  void applyDepthLevels();
                }}
                disabled={applyingLevels}
                className="flex h-11 w-full items-center justify-between rounded-[9px] border border-[#101318] bg-white px-4 text-xs font-medium text-[#101318] transition-colors hover:bg-[#f1f1ed] active:scale-[0.99] disabled:cursor-wait disabled:opacity-60"
              >
                {applyingLevels ? "Applying adjustments…" : "Apply now"}
                <SlidersHorizontal size={16} />
              </button>
              <p className="mt-2 text-[10px] leading-4 text-[#85888d]">
                미리보기는 슬라이더와 동시에 바뀌며, 다운로드 파일은 조작을 멈춘 뒤 자동 저장됩니다.
              </p>
            </div>
          )}
        </fieldset>

        <div className="mt-6 border-t border-[#d9dad6]">
          <Toggle label="Person matte" detail="인물 경계 소스" checked={matte} onChange={setMatte} disabled={Boolean(job)} />
          <Toggle label="Green screen" detail="녹색 배경 합성 소스" checked={alpha} onChange={setAlpha} disabled={Boolean(job)} />
        </div>

        {job && (
          <div className="mt-6" aria-live="polite">
            <div className="flex items-center justify-between text-xs">
              <span>{job.status === "failed" ? "처리 실패" : job.stage}</span>
              <span className="font-mono text-[10px]">{job.progress}%</span>
            </div>
            <div className="mt-2 h-1 overflow-hidden rounded-full bg-[#e2e3df]">
              <div className="h-full bg-[#101318] transition-[width] duration-500" style={{ width: `${job.progress}%` }} />
            </div>
            {job.error && <p className="mt-3 text-xs text-[#b33c32]">{job.error}</p>}
          </div>
        )}
        {error && <p className="mt-3 text-xs leading-5 text-[#b33c32]">{error}</p>}

        {job?.status === "complete" && job.results ? (
          <div className="mt-6 border-t border-[#d9dad6] pt-5">
            <div className="mb-3 flex items-center justify-between">
              <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-[#74777c]">Downloads</p>
              <DownloadSimple size={16} />
            </div>
            <div className="grid gap-2">
              <DownloadLink href={job.results.depth} label="Download Depth" filename="depdy_depth.mp4" version={downloadVersion} />
              {job.results.matte && <DownloadLink href={job.results.matte} label="Download Person Matte" filename="depdy_person_matte.mp4" version={downloadVersion} secondary />}
              {job.results.alpha && <DownloadLink href={job.results.alpha} label="Download Green Screen" filename="depdy_green_screen.mp4" version={downloadVersion} secondary />}
              <DownloadLink href={job.results.validation} label="Download Validation Sheet" filename="depdy_validation_sheet.jpg" version={downloadVersion} secondary />
            </div>
          </div>
        ) : (
        <button
          onClick={startExtraction}
          disabled={job?.status === "queued" || job?.status === "processing"}
          className="mt-6 flex h-12 w-full items-center justify-between rounded-[10px] bg-[#101318] px-4 text-sm font-medium text-white transition-colors hover:bg-[#292d33] active:scale-[0.99] disabled:cursor-wait disabled:bg-[#55595f]"
        >
          {job?.status === "queued" || job?.status === "processing" ? "Extracting depth" : "Extract sources"}
          <ArrowRight size={17} />
        </button>
        )}
        <p className="mt-3 text-center font-mono text-[9px] uppercase tracking-[0.08em] text-[#92959a]">
          파일은 처리 후 24시간 이내 삭제됩니다
        </p>
      </div>
    </div>
  );
}

function LevelSlider({ label, value, min, max, step, display, onChange }: { label: string; value: number; min: number; max: number; step: number; display: string; onChange: (value: number) => void }) {
  return (
    <label className="mt-5 block">
      <span className="flex items-center justify-between text-xs text-[#676b72]">
        <span>{label}</span>
        <span className="font-mono text-[#101318]">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-3 h-1 w-full cursor-pointer accent-[#5f6fff]"
      />
    </label>
  );
}

function DownloadLink({ href, label, filename, version, secondary = false }: { href: string; label: string; filename: string; version: number; secondary?: boolean }) {
  return (
    <a
      href={`${WORKER_URL}${href}?v=${version}`}
      download={filename}
      className={`flex h-12 w-full items-center justify-between rounded-[10px] border px-4 text-sm font-medium transition-colors active:scale-[0.99] ${secondary ? "border-[#c8c9c5] bg-white text-[#101318] hover:bg-[#f3f3f0]" : "border-[#101318] bg-[#101318] text-white hover:bg-[#292d33]"}`}
    >
      {label}
      <DownloadSimple size={17} />
    </a>
  );
}

function Toggle({ label, detail, checked, onChange, disabled = false }: { label: string; detail: string; checked: boolean; onChange: (value: boolean) => void; disabled?: boolean }) {
  return (
    <button type="button" disabled={disabled} onClick={() => onChange(!checked)} className="flex w-full items-center justify-between border-b border-[#d9dad6] py-4 text-left disabled:cursor-not-allowed disabled:opacity-55">
      <span>
        <span className="block text-sm font-medium">{label}</span>
        <span className="mt-1 block text-xs text-[#7b7e83]">{detail}</span>
      </span>
      <span className={`grid size-6 place-items-center rounded-[7px] border ${checked ? "border-[#101318] bg-[#101318] text-white" : "border-[#c8c9c5] bg-white"}`}>
        {checked && <Check size={13} weight="bold" />}
      </span>
    </button>
  );
}
