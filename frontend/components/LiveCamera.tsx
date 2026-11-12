"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ProbabilityBars from "./ProbabilityBars";
import { emotionColor, Face, largestFace, predict, PredictResponse, UNSURE_COLOR } from "@/lib/api";
import { clearCanvas, drawFaces } from "@/lib/draw";

const FRAME_INTERVAL_MS = 500;
const JPEG_QUALITY = 0.8;

type Status = "idle" | "starting" | "running";

function cameraErrorMessage(err: unknown): string {
  const name = (err as DOMException)?.name;
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera access was denied. Allow camera permission for this site in your browser's address bar, then press Start camera again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found on this device.";
    case "NotReadableError":
      return "The camera is already in use by another app or tab. Close it and try again.";
    default:
      return `Could not start the camera: ${(err as Error)?.message || String(err)}`;
  }
}

export default function LiveCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement | null>(null); // hidden, un-mirrored
  const streamRef = useRef<MediaStream | null>(null);
  // Bumped on every start/stop; a loop iteration whose session id is stale exits.
  const sessionRef = useRef(0);
  const lastResultRef = useRef<PredictResponse | null>(null);

  const [status, setStatus] = useState<Status>("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  const redraw = useCallback(() => {
    const canvas = overlayRef.current;
    const res = lastResultRef.current;
    if (!canvas) return;
    if (!res || !res.image_width || !res.image_height) return clearCanvas(canvas);
    drawFaces(canvas, res.faces, {
      sourceWidth: res.image_width,
      sourceHeight: res.image_height,
      mirror: true,
    });
  }, []);

  const stopCamera = useCallback(() => {
    sessionRef.current++; // cancels the capture loop and ignores in-flight responses
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    lastResultRef.current = null;
    clearCanvas(overlayRef.current);
    setStatus("idle");
    setResult(null);
    setLatency(null);
    setRequestError(null);
  }, []);

  /** Grab the current video frame as a JPEG blob (not mirrored). */
  const captureFrame = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return null;
    const canvas = (captureRef.current ??= document.createElement("canvas"));
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
  }, []);

  /**
   * Capture -> send -> wait for the response -> wait out the rest of the
   * interval -> repeat. Because each step awaits the previous one, there is
   * never more than one request in flight.
   */
  const runLoop = useCallback(
    async (session: number) => {
      while (sessionRef.current === session) {
        const started = performance.now();
        try {
          const frame = await captureFrame();
          if (frame) {
            const res = await predict(frame, "frame.jpg");
            if (sessionRef.current !== session) return; // stopped meanwhile
            setLatency(Math.round(performance.now() - started));
            setRequestError(null);
            lastResultRef.current = res;
            setResult(res);
            redraw();
          }
        } catch (err) {
          if (sessionRef.current !== session) return;
          setRequestError((err as Error).message);
          lastResultRef.current = null;
          setResult(null);
          clearCanvas(overlayRef.current);
        }
        const wait = Math.max(0, FRAME_INTERVAL_MS - (performance.now() - started));
        await new Promise((r) => setTimeout(r, wait));
      }
    },
    [captureFrame, redraw],
  );

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setRequestError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "Camera access isn't available. Use a modern browser and open the app via http://localhost (or HTTPS).",
      );
      return;
    }
    const session = ++sessionRef.current;
    setStatus("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (sessionRef.current !== session) {
        // Stopped/unmounted while the permission prompt was open.
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      setStatus("running");
      runLoop(session);
    } catch (err) {
      if (sessionRef.current !== session) return;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setStatus("idle");
      setCameraError(cameraErrorMessage(err));
    }
  }, [runLoop]);

  // Stop the camera when the component unmounts (tab switch) or the page is left.
  useEffect(() => {
    const onPageHide = () => stopCamera();
    window.addEventListener("pagehide", onPageHide);
    return () => {
      window.removeEventListener("pagehide", onPageHide);
      stopCamera();
    };
  }, [stopCamera]);

  // Keep boxes aligned when the video's displayed size changes.
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [redraw]);

  const running = status === "running";
  const main: Face | undefined = result ? largestFace(result.faces) : undefined;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          {status === "idle" ? (
            <button onClick={startCamera} className="btn-primary">
              Start camera
            </button>
          ) : (
            <button onClick={stopCamera} className="btn-secondary">
              {status === "starting" ? "Cancel" : "Stop camera"}
            </button>
          )}
          {running && (
            <span className="inline-flex items-center gap-2 text-sm text-muted">
              <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" /> Live
              {latency !== null && (
                <span className="rounded-md bg-track px-2 py-0.5 font-mono text-xs tabular-nums">
                  {latency} ms
                </span>
              )}
            </span>
          )}
        </div>

        {cameraError && <div className="alert-error">{cameraError}</div>}
        {requestError && <div className="alert-error">{requestError}</div>}

        {/* While running, the video's natural aspect ratio sets the box height,
            so the overlay canvas (inset-0) covers exactly the displayed video. */}
        <div
          className={`relative overflow-hidden rounded-2xl border border-line bg-black ${running ? "" : "aspect-video"}`}
        >
          {/* Mirrored like a selfie camera; the overlay canvas is NOT mirrored,
              boxes are mirrored in code instead so their labels stay readable. */}
          <video
            ref={videoRef}
            playsInline
            muted
            className={running ? "block h-auto w-full -scale-x-100" : "hidden"}
          />
          <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          {!running && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-white/70">
              <CameraIcon />
              {status === "starting" ? "Waiting for camera permission…" : "Camera is off"}
            </div>
          )}
        </div>
      </section>

      <aside className="card h-fit space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          {result && result.faces.length > 1 ? "Largest face" : "Expression"}
        </h2>
        {!running ? (
          <p className="text-sm text-muted">Start the camera to see live predictions.</p>
        ) : !result ? (
          <p className="text-sm text-muted">Analyzing…</p>
        ) : !main ? (
          <p className="text-base font-medium">No face detected</p>
        ) : (
          <>
            <div>
              <p
                className="text-3xl font-bold capitalize"
                style={{ color: main.sure ? emotionColor(main.emotion) : UNSURE_COLOR }}
              >
                {main.sure ? main.emotion : "Not sure"}
              </p>
              <p className="text-sm text-muted">
                {main.sure
                  ? `${(main.confidence * 100).toFixed(1)}% confidence`
                  : `Best guess: ${main.emotion} (${(main.confidence * 100).toFixed(1)}%)`}
                {result.faces.length > 1 && ` · ${result.faces.length} faces`}
              </p>
            </div>
            <ProbabilityBars probabilities={main.probabilities} />
          </>
        )}
      </aside>
    </div>
  );
}

function CameraIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path d="M15 10l4.55-2.28A1 1 0 0121 8.62v6.76a1 1 0 01-1.45.9L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  );
}
