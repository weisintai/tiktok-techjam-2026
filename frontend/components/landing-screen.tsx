"use client";

import { ArrowRight, PackageSearch } from "lucide-react";

import { Button } from "@/components/ui/button";

type LandingScreenProps = {
  onStart: () => void;
};

export function LandingScreen({ onStart }: LandingScreenProps) {
  return (
    <main className="dashboard-grid relative flex h-screen min-h-0 flex-col items-center justify-center overflow-hidden px-4">
      <div className="flex w-full max-w-2xl flex-col items-center gap-8 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-md bg-primary text-primary-foreground shadow-soft">
          <PackageSearch className="h-8 w-8" aria-hidden="true" />
        </div>

        <div className="space-y-3">
          <p className="text-sm font-semibold uppercase tracking-wide text-primary">
            TikTok TechJam 2026
          </p>
          <h1 className="text-4xl font-bold tracking-normal text-foreground sm:text-5xl">
            Shopping Copilot
          </h1>
          <p className="mx-auto max-w-xl text-base text-muted-foreground sm:text-lg">
            An offline conversational agent that turns a chat into typed
            constraints and finds the exact product you want, without losing
            track of anything you already said.
          </p>
        </div>

        <Button className="h-12 px-8 text-base" onClick={onStart}>
          Start Shopping
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </main>
  );
}
