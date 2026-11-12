"use client";

import { useState } from "react";
import BackendStatus from "@/components/BackendStatus";
import LiveCamera from "@/components/LiveCamera";
import UploadPhoto from "@/components/UploadPhoto";

type Tab = "live" | "upload";

const TABS: { id: Tab; label: string }[] = [
  { id: "live", label: "Live camera" },
  { id: "upload", label: "Upload photo" },
];

export default function Home() {
  const [tab, setTab] = useState<Tab>("live");

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 lg:py-12">
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Expression Detector</h1>
          <p className="mt-1 text-muted">
            Facial expression recognition with a ResNet18 trained on FER-2013.
          </p>
        </div>
        <BackendStatus />
      </header>

      <div role="tablist" aria-label="Mode" className="mb-6 inline-flex rounded-xl bg-track p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              tab === t.id ? "bg-surface shadow-sm" : "text-muted hover:text-foreground"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* LiveCamera unmounts when hidden, which stops the camera.
          UploadPhoto stays mounted (just hidden) so its photo/results survive tab switches. */}
      {tab === "live" && <LiveCamera />}
      <div hidden={tab !== "upload"}>
        <UploadPhoto />
      </div>

      <p className="mt-10 rounded-xl border border-line bg-surface px-4 py-3 text-sm text-muted">
        <strong className="text-foreground">Note:</strong> This app detects facial expressions, not
        actual feelings. Frames are processed in memory and never stored.
      </p>
    </main>
  );
}
