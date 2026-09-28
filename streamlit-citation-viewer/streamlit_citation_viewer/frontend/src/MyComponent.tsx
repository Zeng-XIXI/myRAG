import { FC, MouseEvent, ReactElement, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Source = {
  source: string;
  chunk: number;
  distance: number;
  text: string;
  start_line?: number;
  end_line?: number;
  file_content?: string;
  file_available?: boolean;
  file_current?: boolean;
  is_markdown?: boolean;
};

export type CitationViewerData = {
  answer: string;
  sources: Source[];
};

type Popup = {
  sourceNumber: number;
  top: number;
  left: number;
};

const citationUrl = "#rag-citation-";

function linkifyCitations(answer: string): string {
  return answer.replace(/\[(\d+)\]/g, (_match, number) =>
    `[[${number}]](${citationUrl}${number})`,
  );
}

function SourcePreview({ source }: { source: Source }): ReactElement {
  const [mode, setMode] = useState<"location" | "preview">("location");
  const highlightedLine = useRef<HTMLDivElement | null>(null);
  const lines = (source.file_content ?? source.text).split(/\r?\n/);
  const startLine = source.start_line ?? 1;
  const endLine = source.end_line ?? startLine;

  useEffect(() => {
    if (mode === "location") {
      highlightedLine.current?.scrollIntoView({ block: "center" });
    }
  }, [mode, source.source, startLine]);

  return (
    <section className="citation-preview">
      <header className="preview-header">
        <div>
          <strong>{source.source}</strong>
          <span>
            片段 {source.chunk}
            {source.start_line ? ` · 第 ${startLine}–${endLine} 行` : ""}
            {Number.isFinite(source.distance) ? ` · 距离 ${source.distance.toFixed(4)}` : ""}
          </span>
        </div>
      </header>

      {!source.file_available && (
        <p className="preview-warning">原文件当前不存在，以下显示入库时的片段。</p>
      )}
      {source.file_available && source.file_current === false && (
        <p className="preview-warning">原文件已发生变化，请同步知识库后再核对行号。</p>
      )}

      <nav className="preview-tabs" aria-label="预览方式">
        <button
          className={mode === "location" ? "active" : ""}
          onClick={() => setMode("location")}
          type="button"
        >
          定位原文
        </button>
        {source.is_markdown && source.file_available && (
          <button
            className={mode === "preview" ? "active" : ""}
            onClick={() => setMode("preview")}
            type="button"
          >
            Markdown 预览
          </button>
        )}
      </nav>

      <div className="preview-body">
        {mode === "preview" ? (
          <article className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {source.file_content ?? ""}
            </ReactMarkdown>
          </article>
        ) : (
          <div className="source-lines">
            {lines.map((line, index) => {
              const lineNumber = index + 1;
              const highlighted = lineNumber >= startLine && lineNumber <= endLine;
              return (
                <div
                  className={highlighted ? "source-line highlighted" : "source-line"}
                  key={lineNumber}
                  ref={lineNumber === startLine ? highlightedLine : undefined}
                >
                  <span className="line-number">{lineNumber}</span>
                  <span className="line-text">{line || " "}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
      <div className="resize-hint">拖动右下角可调整大小</div>
    </section>
  );
}

const CitationViewer: FC<CitationViewerData> = ({ answer, sources }): ReactElement => {
  const [popup, setPopup] = useState<Popup | null>(null);
  const linkedAnswer = useMemo(() => linkifyCitations(answer), [answer]);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPopup(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  const openCitation = (event: MouseEvent<HTMLButtonElement>, sourceNumber: number) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const popupWidth = 560;
    const popupHeight = 440;
    setPopup({
      sourceNumber,
      top: Math.max(12, Math.min(rect.top, window.innerHeight - popupHeight - 12)),
      left: Math.max(12, Math.min(rect.left, window.innerWidth - popupWidth - 12)),
    });
  };

  return (
    <div className="citation-answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith(citationUrl)) {
              const sourceNumber = Number(href.slice(citationUrl.length));
              const valid = sourceNumber >= 1 && sourceNumber <= sources.length;
              return (
                <button
                  aria-label={`预览引用 ${sourceNumber}`}
                  className="citation-link"
                  disabled={!valid}
                  onClick={(event) => valid && openCitation(event, sourceNumber)}
                  title={valid ? `预览引用 ${sourceNumber}` : "引用来源不存在"}
                  type="button"
                >
                  {children}
                </button>
              );
            }
            return <a href={href}>{children}</a>;
          },
        }}
      >
        {linkedAnswer}
      </ReactMarkdown>

      {popup && (
        <div
          className="citation-popup"
          style={{ top: popup.top, left: popup.left }}
        >
          <button
            aria-label="关闭引用预览"
            className="close-button"
            onClick={() => setPopup(null)}
            type="button"
          >
            ×
          </button>
          <SourcePreview source={sources[popup.sourceNumber - 1]} />
        </div>
      )}
    </div>
  );
};

export default CitationViewer;
