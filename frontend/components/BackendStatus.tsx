"use client";

import { useEffect, useState } from "react";
import { API_URL, checkHealth } from "@/lib/api";

type State = "checking" | "online" | "offline";

/** Small pill showing whether the FastAPI backend is reachable (re-checks every 10 s). */
export default function BackendStatus() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      checkHealth()
        .then(() => !cancelled && setState("online"))
        .catch(() => !cancelled && setState("offline"));
    check();
    const id = setInterval(check, 10_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const styles: Record<State, string> = {
    checking: "bg-track text-muted",
    online: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
    offline: "bg-red-500/15 text-red-700 dark:text-red-300",
  };
  const dot: Record<State, string> = {
    checking: "bg-gray-400",
    online: "bg-emerald-500",
    offline: "bg-red-500",
  };
  const text: Record<State, string> = {
    checking: "Checking backend…",
    online: "Backend online",
    offline: "Backend unreachable",
  };

  return (
    <span
      title={state === "offline" ? `Start the FastAPI server at ${API_URL}` : API_URL}
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${styles[state]}`}
    >
      <span className={`h-2 w-2 rounded-full ${dot[state]}`} />
      {text[state]}
    </span>
  );
}
