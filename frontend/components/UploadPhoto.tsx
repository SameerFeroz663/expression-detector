"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ProbabilityBars from "./ProbabilityBars";
import { emotionColor, MAX_UPLOAD_BYTES, predict, PredictResponse, UNSURE_COLOR } from "@/lib/api";
import { clearCanvas, drawFaces } from "@/lib/draw";

function formatBytes(n: number) {
  return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function UploadPhoto() {
  const inputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // Free the object URL when the preview changes or the component unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const selectFile = useCallback((f: File | undefined) => {
    setError(null);
    setResult(null);
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file (JPEG, PNG, WebP…).");
      return;
    }
    if (f.size > MAX_UPLOAD_BYTES) {
      setError(`That image is ${formatBytes(f.size)}. The maximum is 5 MB.`);
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  }, []);

  const redraw = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    if (!result?.image_width || !result.image_height) return clearCanvas(canvas);
    drawFaces(canvas, result.faces, {
      sourceWidth: result.image_width,
      sourceHeight: result.image_height,
      numbered: true,
    });
  }, [result]);

  // Redraw when results arrive and whenever the preview is resized.
  useEffect(() => {
    redraw();
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [redraw, previewUrl]);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(file, file.name));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <section className="space-y-4">
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => selectFile(e.target.files?.[0])}
        />

        {!previewUrl ? (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              selectFile(e.dataTransfer.files?.[0]);
            }}
            className={`flex aspect-video w-full flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-6 text-center transition-colors ${
              dragging ? "border-accent bg-accent/10" : "border-line bg-surface hover:border-accent/60"
            }`}
          >
            <UploadIcon />
            <span className="font-medium">Drop a photo here, or click to choose</span>
            <span className="text-sm text-muted">Images only · max 5 MB</span>
          </button>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <button onClick={analyze} disabled={loading} className="btn-primary">
                {loading ? "Analyzing…" : result ? "Analyze again" : "Analyze"}
              </button>
              <button onClick={reset} disabled={loading} className="btn-secondary">
                Choose another photo
              </button>
              {file && (
                <span className="truncate text-sm text-muted">
                  {file.name} · {formatBytes(file.size)}
                </span>
              )}
            </div>
            <div className="flex justify-center rounded-2xl border border-line bg-surface p-2">
              {/* inline-block wrapper shrinks to the image so the canvas matches it exactly */}
              <div className="relative inline-block max-w-full">
                {/* eslint-disable-next-line @next/next/no-img-element -- local blob preview */}
                <img
                  ref={imgRef}
                  src={previewUrl}
                  alt="Uploaded preview"
                  onLoad={redraw}
                  className="block h-auto max-h-[70vh] max-w-full rounded-lg"
                />
                <canvas
                  ref={overlayRef}
                  className="pointer-events-none absolute inset-0 h-full w-full"
                />
              </div>
            </div>
          </>
        )}

        {error && <div className="alert-error">{error}</div>}
      </section>

      <aside className="card h-fit space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Results</h2>
        {!result ? (
          <p className="text-sm text-muted">
            {previewUrl ? "Press Analyze to detect faces." : "Upload a photo to get started."}
          </p>
        ) : result.faces.length === 0 ? (
          <p className="text-base font-medium">No face detected</p>
        ) : (
          <ol className="space-y-6">
            {result.faces.map((face, i) => (
              <li key={i} className="space-y-3 border-t border-line pt-4 first:border-0 first:pt-0">
                <div className="flex items-baseline justify-between gap-2">
                  <p className="font-semibold">
                    <span className="mr-2 text-muted">#{i + 1}</span>
                    <span
                      className="capitalize"
                      style={{ color: face.sure ? emotionColor(face.emotion) : UNSURE_COLOR }}
                    >
                      {face.emotion}
                    </span>
                    {!face.sure && <span className="ml-2 text-xs font-normal text-muted">(not sure)</span>}
                  </p>
                  <span className="tabular-nums text-sm text-muted">
                    {(face.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <ProbabilityBars probabilities={face.probabilities} />
              </li>
            ))}
          </ol>
        )}
      </aside>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted" aria-hidden>
      <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12M7 9l5-5 5 5" />
    </svg>
  );
}
