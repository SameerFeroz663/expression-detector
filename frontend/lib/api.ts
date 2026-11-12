// Shared types + API helpers for talking to the FastAPI backend.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(
  /\/+$/,
  "",
);

export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024; // keep in sync with the backend

export type Probability = { label: string; probability: number };

export type Face = {
  box: [number, number, number, number]; // x, y, w, h in original image pixels
  emotion: string;
  confidence: number;
  sure: boolean;
  probabilities: Probability[]; // sorted high -> low
};

export type PredictResponse = {
  image_width?: number;
  image_height?: number;
  faces: Face[];
};

export class BackendUnreachableError extends Error {
  constructor() {
    super(
      `Can't reach the backend at ${API_URL}. Make sure the FastAPI server is running (uvicorn main:app --port 8000).`,
    );
    this.name = "BackendUnreachableError";
  }
}

/** POST an image (File or Blob) to /predict and return the parsed result. */
export async function predict(image: Blob, filename = "image.jpg", signal?: AbortSignal) {
  const form = new FormData();
  form.append("file", image, filename);

  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, { method: "POST", body: form, signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    // fetch only rejects on network failures (server down, CORS, wrong URL).
    throw new BackendUnreachableError();
  }

  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") message = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(message);
  }
  return (await res.json()) as PredictResponse;
}

export async function checkHealth(): Promise<{ status: string; classes: string[] }> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  } catch {
    throw new BackendUnreachableError();
  }
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Emotion colors (class names come from the model, so unknown names fall back
// to a palette based on a hash of the name).
// ---------------------------------------------------------------------------
const EMOTION_COLORS: Record<string, string> = {
  angry: "#ef4444",
  disgust: "#22c55e",
  fear: "#a855f7",
  happy: "#f59e0b",
  neutral: "#94a3b8",
  sad: "#3b82f6",
  surprise: "#ec4899",
};
const FALLBACK = ["#14b8a6", "#f97316", "#84cc16", "#06b6d4", "#e11d48", "#8b5cf6"];

export function emotionColor(label: string): string {
  const known = EMOTION_COLORS[label.toLowerCase()];
  if (known) return known;
  let h = 0;
  for (const ch of label) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return FALLBACK[h % FALLBACK.length];
}

export const UNSURE_COLOR = "#9ca3af";

export function faceLabel(face: Face): string {
  return face.sure ? `${face.emotion} ${Math.round(face.confidence * 100)}%` : "not sure";
}

export function largestFace(faces: Face[]): Face | undefined {
  return faces.reduce<Face | undefined>(
    (best, f) => (!best || f.box[2] * f.box[3] > best.box[2] * best.box[3] ? f : best),
    undefined,
  );
}
