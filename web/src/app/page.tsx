import { InstagramLogo } from "@phosphor-icons/react/dist/ssr";
import { DepthWorkspace } from "@/components/depth-workspace";

export default function Home() {
  return (
    <main className="bg-grain min-h-[100dvh] text-[#101318]">
      <header className="mx-auto flex h-20 w-full max-w-[1440px] items-center justify-between px-5 md:px-8">
        <a href="#top" className="text-[17px] font-semibold tracking-[-0.04em]">
          DEPDY
        </a>
        <div className="flex items-center gap-5 font-mono text-[11px] uppercase tracking-[0.12em] text-[#62666d]">
          <span className="hidden sm:inline">Local GPU</span>
          <span>KR</span>
        </div>
      </header>

      <section id="top" className="mx-auto grid w-full max-w-[1440px] gap-10 px-5 pb-16 pt-10 md:px-8 lg:grid-cols-[0.75fr_1.25fr] lg:gap-16 lg:pb-24 lg:pt-16">
        <div className="flex flex-col justify-between gap-10 lg:min-h-[620px]">
          <div>
            <h1 className="max-w-[650px] text-[clamp(3.4rem,7vw,7.6rem)] font-medium leading-[0.88] tracking-[-0.075em]">
              Motion into depth.
            </h1>
            <p className="mt-8 max-w-md text-base leading-7 text-[#666a70]">
              <span className="block">영상의 움직임은 그대로 두고,</span>
              <span className="block">Depth와 인물 소스를 정교하게 분리합니다.</span>
            </p>
          </div>
          <div className="grid grid-cols-3 border-t border-[#cacbc7] pt-4 font-mono text-[10px] uppercase leading-5 tracking-[0.1em] text-[#72767b]">
            <span>1080P</span>
            <span>60 SEC</span>
            <span className="text-right">24H DELETE</span>
          </div>
        </div>

        <DepthWorkspace />
      </section>

      <section className="border-t border-[#cacbc7]">
        <div className="mx-auto grid max-w-[1440px] gap-12 px-5 py-20 md:px-8 lg:grid-cols-2 lg:py-28">
          <h2 className="max-w-lg text-4xl font-medium leading-[1.03] tracking-[-0.055em] md:text-6xl">
            <span className="block">One clip.</span>
            <span className="block">Every</span>
            <span className="block">control source.</span>
          </h2>
          <div className="grid gap-0 border-t border-[#cacbc7]">
            {[
              ["01", "Relative Depth", "장면의 앞뒤 구조와 움직임을 보존합니다."],
              ["02", "Person Matte", "머리카락과 빠른 동작의 경계를 분리합니다."],
              ["03", "Green Screen Source", "인물은 유지하고 배경을 녹색으로 합성한 소스를 만듭니다."],
              ["04", "Line Art", "검정 배경에 흰색 아웃라인만 남깁니다."],
              ["05", "Pose Skeleton", "ControlNet OpenPose 스타일 스켈레톤을 추출합니다."],
              ["06", "Seedance Pack", "생성형 영상 작업에 필요한 파일을 묶어냅니다."],
            ].map(([number, title, description]) => (
              <div key={number} className="grid grid-cols-[44px_1fr] gap-3 border-b border-[#cacbc7] py-5 sm:grid-cols-[54px_0.8fr_1.2fr]">
                <span className="font-mono text-[10px] text-[#888b90]">{number}</span>
                <strong className="text-sm font-medium">{title}</strong>
                <p className="col-start-2 text-sm leading-6 text-[#6d7075] sm:col-start-auto">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="mx-auto flex max-w-[1440px] items-center justify-between px-5 pb-8 pt-14 md:px-8">
        <a
          href="https://www.instagram.com/from.leella"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm font-medium tracking-[-0.02em] text-[#101318] transition-colors hover:text-[#62666d]"
        >
          <InstagramLogo size={18} weight="regular" />
          @from.leella
        </a>
        <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-[#777a80]">Private local processing</p>
      </footer>
    </main>
  );
}
