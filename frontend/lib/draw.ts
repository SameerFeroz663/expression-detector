import { emotionColor, Face, faceLabel, UNSURE_COLOR } from "./api";

type DrawOptions = {
  /** Resolution of the image the backend analysed (box coordinates live here). */
  sourceWidth: number;
  sourceHeight: number;
  /** Mirror boxes horizontally (for a mirrored selfie video). */
  mirror?: boolean;
  /** Prefix labels with "#1", "#2", ... so they match a results list. */
  numbered?: boolean;
};

/**
 * Draw face boxes onto an overlay canvas that sits exactly on top of the
 * displayed image/video. The canvas is sized to its CSS size (x devicePixelRatio
 * for sharpness) and box coordinates are scaled from the source resolution.
 */
export function drawFaces(canvas: HTMLCanvasElement, faces: Face[], opts: DrawOptions) {
  const dpr = window.devicePixelRatio || 1;
  const displayW = canvas.clientWidth;
  const displayH = canvas.clientHeight;
  if (canvas.width !== Math.round(displayW * dpr) || canvas.height !== Math.round(displayH * dpr)) {
    canvas.width = Math.round(displayW * dpr);
    canvas.height = Math.round(displayH * dpr);
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, displayW, displayH);
  if (!opts.sourceWidth || !opts.sourceHeight) return;

  const sx = displayW / opts.sourceWidth;
  const sy = displayH / opts.sourceHeight;
  const fontSize = Math.max(12, Math.min(18, displayW / 40));
  ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = "top";

  faces.forEach((face, i) => {
    const [x, y, w, h] = face.box;
    // Mirror in source coordinates first (x' = width - x - w), then scale.
    const srcX = opts.mirror ? opts.sourceWidth - x - w : x;
    const dx = srcX * sx;
    const dy = y * sy;
    const dw = w * sx;
    const dh = h * sy;

    const color = face.sure ? emotionColor(face.emotion) : UNSURE_COLOR;
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.setLineDash(face.sure ? [] : [6, 4]);
    ctx.strokeRect(dx, dy, dw, dh);
    ctx.setLineDash([]);

    const text = (opts.numbered ? `#${i + 1} ` : "") + faceLabel(face);
    const padX = 6;
    const labelH = fontSize + 8;
    const labelW = ctx.measureText(text).width + padX * 2;
    // Put the label above the box, or inside it if there's no room above.
    const ly = dy - labelH >= 0 ? dy - labelH : dy;
    const lx = Math.min(Math.max(0, dx - 1.5), Math.max(0, displayW - labelW));
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, labelW, labelH);
    ctx.fillStyle = "#0b0b0f";
    ctx.fillText(text, lx + padX, ly + 4);
  });
}

export function clearCanvas(canvas: HTMLCanvasElement | null) {
  canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
}
