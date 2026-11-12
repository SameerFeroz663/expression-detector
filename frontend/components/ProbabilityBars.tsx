import { emotionColor, Probability } from "@/lib/api";

/** Horizontal bars for every emotion, highest probability first. */
export default function ProbabilityBars({ probabilities }: { probabilities: Probability[] }) {
  const sorted = [...probabilities].sort((a, b) => b.probability - a.probability);
  return (
    <ul className="space-y-2">
      {sorted.map(({ label, probability }) => {
        const pct = probability * 100;
        return (
          <li key={label}>
            <div className="mb-1 flex justify-between text-sm">
              <span className="capitalize">{label}</span>
              <span className="tabular-nums text-muted">{pct.toFixed(1)}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-track">
              <div
                className="h-full rounded-full transition-[width] duration-300 ease-out"
                style={{ width: `${pct}%`, backgroundColor: emotionColor(label) }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
