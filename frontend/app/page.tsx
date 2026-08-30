"use client";

import { FormEvent, useState } from "react";
import {
  BadgeCheck,
  BrainCircuit,
  ChevronRight,
  MessageSquareText,
  PackageSearch,
  RefreshCw,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LandingScreen } from "@/components/landing-screen";
import { metrics, quickPrompts } from "@/lib/demo-data";

type LiveProduct = {
  asin: string;
  rank: number;
  title: string;
  category: string;
  price: number | null;
  rating: number | null;
  ratingCount: number | null;
  store: string;
};

type LiveState = {
  category: string;
  constraints: string[];
  slots: Record<string, string[]>;
  negativeConstraints: string[];
  intent: string;
  softQueries: string[];
  seenCount: number;
};

type LiveMessage = {
  role: "shopper" | "copilot";
  text: string;
  turn: number;
};

const starterPrompt = "I am looking for t-shirts. A key requirement is: 67% Polyester, 33% Cotton.";

const starterMessage: LiveMessage = {
  role: "copilot",
  turn: 0,
  text: "Tell me what you are shopping for. I will narrow the catalog as you add preferences.",
};

const shopperSteps = [
  { label: "Ask", value: "Describe the item" },
  { label: "Refine", value: "Add fit, material, color, or budget" },
  { label: "Adjust", value: "Change your mind mid-session" },
  { label: "Review", value: "Compare ranked matches" },
];

export default function Home() {
  const [showLanding, setShowLanding] = useState(true);
  const [input, setInput] = useState(starterPrompt);
  const [sessionId, setSessionId] = useState(() => `demo-${Date.now()}`);
  const [turn, setTurn] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [liveMessages, setLiveMessages] = useState<LiveMessage[]>([]);
  const [liveProducts, setLiveProducts] = useState<LiveProduct[]>([]);
  const [liveState, setLiveState] = useState<LiveState | null>(null);

  const displayMessages = liveMessages.length ? liveMessages : [starterMessage];
  const liveSlots = liveState?.slots ?? {};
  const liveNegatives = liveState?.negativeConstraints ?? [];
  const sessionMode = liveState
    ? liveState.intent !== "unknown"
      ? liveState.intent
      : liveState.constraints.length > 0
        ? "narrowing"
        : "listening"
    : "listening";

  async function sendMessage(nextMessage?: string) {
    const message = (nextMessage ?? input).trim();
    if (!message || isLoading) return;

    setIsLoading(true);
    setError("");
    setLiveMessages((items) => [...items, { role: "shopper", text: message, turn }]);
    setInput("");

    try {
      const response = await fetch("/api/copilot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: "respond", sessionId, message, turn }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error ?? "Backend request failed");
      }
      setLiveMessages((items) => [
        ...items,
        { role: "copilot", text: payload.message, turn },
      ]);
      setLiveProducts(payload.recommendations ?? []);
      setLiveState(payload.state);
      setTurn((value) => value + 1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong");
    } finally {
      setIsLoading(false);
    }
  }

  async function resetLiveDemo() {
    const nextSessionId = `demo-${Date.now()}`;
    setSessionId(nextSessionId);
    setTurn(1);
    setInput(starterPrompt);
    setLiveMessages([]);
    setLiveProducts([]);
    setLiveState(null);
    setError("");
    await fetch("/api/copilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: "reset", sessionId: nextSessionId }),
    }).catch(() => undefined);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage();
  }

  if (showLanding) {
    return <LandingScreen onStart={() => setShowLanding(false)} />;
  }

  return (
    <main className="flex h-screen min-h-0 flex-col overflow-hidden">
      <section className="dashboard-grid shrink-0 border-b border-border/70">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground shadow-soft">
              <PackageSearch className="h-5 w-5" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold uppercase text-primary">TikTok TechJam 2026</p>
              <h1 className="truncate text-2xl font-bold tracking-normal text-foreground sm:text-3xl">
                Shopping Copilot
              </h1>
            </div>
          </div>
          <div className="hidden flex-wrap items-center gap-2 sm:flex">
            <Badge variant="success" className="gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
              Live catalog
            </Badge>
            <Badge variant="outline" className="gap-1.5">
              <BrainCircuit className="h-3.5 w-3.5" aria-hidden="true" />
              Remembers preferences
            </Badge>
          </div>
        </div>
      </section>

      <section className="mx-auto grid min-h-0 w-full max-w-7xl flex-1 gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[250px_minmax(0,1fr)_300px] lg:px-8">
        <aside className="min-h-0 space-y-3 overflow-hidden">
          <Card className="bg-white/90">
            <CardHeader className="p-4 pb-2">
              <CardTitle>Start Shopping</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 p-4 pt-0">
              {quickPrompts.map((prompt) => (
                <Button
                  key={prompt.label}
                  type="button"
                  variant="outline"
                  className="h-auto w-full justify-between px-3 py-3 text-left"
                  onClick={() => setInput(prompt.value)}
                >
                  <span className="min-w-0 truncate">{prompt.label}</span>
                  <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                </Button>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-white/90">
            <CardHeader className="p-4 pb-2">
              <CardTitle>Shopping Flow</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 p-4 pt-0">
              {shopperSteps.map((step) => (
                <div key={step.label} className="rounded-md border bg-white px-3 py-2">
                  <p className="text-sm font-semibold">{step.label}</p>
                  <p className="mt-0.5 text-xs text-muted-foreground">{step.value}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        </aside>

        <section className="min-h-0">
          <Card className="flex h-full min-h-0 flex-col bg-white/94 shadow-soft">
            <CardHeader className="shrink-0 border-b bg-white/70 p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <MessageSquareText className="h-5 w-5 text-primary" aria-hidden="true" />
                    Shopping Assistant
                  </CardTitle>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Ask naturally, then refine by material, color, style, or budget.
                  </p>
                </div>
                <Badge variant="secondary">50k product catalog</Badge>
              </div>
            </CardHeader>

            <CardContent className="flex min-h-0 flex-1 flex-col gap-4 p-4">
              <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
                {displayMessages.map((message, index) => (
                  <div
                    key={`${message.role}-${message.turn}-${index}`}
                    className={message.role === "shopper" ? "flex justify-end" : "flex justify-start"}
                  >
                    <div
                      className={
                        message.role === "shopper"
                          ? "max-w-[86%] rounded-lg bg-primary px-4 py-3 text-sm leading-6 text-primary-foreground"
                          : "max-w-[86%] rounded-lg bg-muted px-4 py-3 text-sm leading-6 text-foreground"
                      }
                    >
                      <div className="mb-1 text-xs font-semibold uppercase opacity-75">
                        {message.role === "shopper" ? "You" : "Copilot"}
                        {message.turn ? ` · Turn ${message.turn}` : ""}
                      </div>
                      {message.text}
                    </div>
                  </div>
                ))}

                <div className="rounded-lg border bg-[#fbfdfb] p-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold">
                        {liveProducts.length === 1 ? "Top Match" : "Recommended Matches"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {liveProducts.length
                          ? `Showing ${liveProducts.length} of up to 10 scored recommendations`
                          : "Results appear after your first message"}
                      </p>
                    </div>
                    <Badge variant="outline">Session: {sessionMode}</Badge>
                  </div>

                  {liveProducts.length ? (
                    <div className="grid gap-3">
                      {liveProducts.map((product, index) => (
                        <article
                          key={product.asin}
                          className="grid gap-3 rounded-md border bg-white p-3 sm:grid-cols-[1fr_auto]"
                        >
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <Badge variant="outline">Rank {index + 1}</Badge>
                              <span className="font-mono text-xs text-muted-foreground">{product.asin}</span>
                            </div>
                            <h2 className="mt-2 text-sm font-semibold tracking-normal sm:text-base">
                              {product.title}
                            </h2>
                            <p className="mt-1 text-xs text-muted-foreground sm:text-sm">
                              {product.category}
                            </p>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {[product.store, "catalog match", "offline ranked"].map((badge) => (
                                <Badge key={badge} variant="secondary">
                                  {badge}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          <div className="flex items-center justify-between gap-5 sm:min-w-28 sm:flex-col sm:items-end">
                            <div className="text-right">
                              <p className="text-xs text-muted-foreground">Rank</p>
                              <p className="text-xl font-bold text-primary">#{product.rank}</p>
                            </div>
                            <div className="flex gap-2 text-sm font-semibold">
                              <span>
                                {typeof product.price === "number"
                                  ? `$${product.price.toFixed(2)}`
                                  : "No price"}
                              </span>
                              <span className="text-muted-foreground">·</span>
                              <span>{product.rating ?? "N/A"}</span>
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className="flex min-h-24 items-center justify-center rounded-md border border-dashed bg-white text-center text-sm text-muted-foreground">
                      Ask for a product and the copilot will return live catalog matches here.
                    </div>
                  )}
                </div>
              </div>

              <form onSubmit={handleSubmit} className="flex shrink-0 flex-col gap-3 border-t pt-4 sm:flex-row">
                <input
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Try: For that, what matters is: comfortable fabric; casual wear."
                  className="h-11 min-w-0 flex-1 rounded-md border bg-white px-3 text-sm outline-none ring-primary/20 transition focus:ring-4"
                />
                <div className="flex gap-2">
                  <Button type="submit" disabled={isLoading || !input.trim()} className="flex-1 sm:flex-none">
                    <Send className="h-4 w-4" aria-hidden="true" />
                    {isLoading ? "Thinking" : "Send"}
                  </Button>
                  <Button type="button" variant="outline" onClick={() => void resetLiveDemo()} aria-label="Reset live demo">
                    <RefreshCw className="h-4 w-4" aria-hidden="true" />
                  </Button>
                </div>
              </form>

              {error ? (
                <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                  {error}
                </p>
              ) : null}
            </CardContent>
          </Card>
        </section>

        <aside className="min-h-0 space-y-3 overflow-hidden">
          <Card className="bg-white/90">
            <CardHeader className="p-4 pb-2">
              <CardTitle className="flex items-center gap-2">
                <BadgeCheck className="h-5 w-5 text-primary" aria-hidden="true" />
                Preference Memory
              </CardTitle>
            </CardHeader>
            <CardContent className="max-h-[35vh] space-y-3 overflow-y-auto p-4 pt-0">
              {Object.keys(liveSlots).length ? (
                Object.entries(liveSlots).map(([slot, values]) => (
                  <div key={slot}>
                    <p className="text-xs font-semibold uppercase text-muted-foreground">{slot}</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {values.map((value) => (
                        <Badge key={`${slot}-${value}`} variant="outline" className="bg-white">
                          {value}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <p className="rounded-md border border-dashed bg-white p-3 text-sm text-muted-foreground">
                  Preferences will appear here as you chat.
                </p>
              )}
              <div>
                <p className="text-xs font-semibold uppercase text-muted-foreground">Avoiding</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {liveNegatives.length ? (
                    liveNegatives.map((value) => (
                      <Badge key={value} variant="warning">
                        {value}
                      </Badge>
                    ))
                  ) : (
                    <Badge variant="outline">none</Badge>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-[#f5fbf8]">
            <CardHeader className="p-4 pb-2">
              <CardTitle className="flex items-center gap-2">
                <SlidersHorizontal className="h-5 w-5 text-primary" aria-hidden="true" />
                Session Status
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3 p-4 pt-0 text-sm">
              <div className="rounded-md border bg-white p-3">
                <p className="text-xs uppercase text-muted-foreground">Seen</p>
                <p className="mt-1 text-xl font-bold">{liveState?.seenCount ?? 0}</p>
              </div>
              <div className="rounded-md border bg-white p-3">
                <p className="text-xs uppercase text-muted-foreground">Tokens</p>
                <p className="mt-1 text-xl font-bold">0</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/90">
            <CardHeader className="p-4 pb-2">
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-primary" aria-hidden="true" />
                Evaluation
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2 p-4 pt-0">
              {metrics.map((metric) => (
                <div key={metric.label} className="rounded-md border bg-white p-2">
                  <p className="text-xs text-muted-foreground">{metric.label}</p>
                  <p className="mt-1 text-lg font-bold">{metric.value}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        </aside>
      </section>
    </main>
  );
}
