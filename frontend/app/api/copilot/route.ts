import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { createInterface } from "node:readline";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeout: NodeJS.Timeout;
};

declare global {
  // eslint-disable-next-line no-var
  var copilotBridge:
    | {
        process: ChildProcessWithoutNullStreams;
        pending: Map<string, PendingRequest>;
      }
    | undefined;
}

let nextId = 1;

function bridge() {
  if (globalThis.copilotBridge && !globalThis.copilotBridge.process.killed) {
    return globalThis.copilotBridge;
  }

  const repoRoot = process.cwd().replace(/\/frontend$/, "");
  const venvPython = path.join(repoRoot, ".venv", "bin", "python");
  const python = process.env.PYTHON ?? (existsSync(venvPython) ? venvPython : "python3");
  const child = spawn(
    /* turbopackIgnore: true */ python,
    ["frontend/backend/copilot_server.py"],
    {
      cwd: repoRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    },
  );
  const pending = new Map<string, PendingRequest>();
  const reader = createInterface({ input: child.stdout });

  reader.on("line", (line) => {
    try {
      const payload = JSON.parse(line) as {
        id?: string;
        result?: unknown;
        error?: string;
      };
      if (!payload.id) return;
      const request = pending.get(payload.id);
      if (!request) return;
      clearTimeout(request.timeout);
      pending.delete(payload.id);
      if (payload.error) {
        request.reject(new Error(payload.error));
      } else {
        request.resolve(payload.result);
      }
    } catch {
      // Ignore non-JSON startup noise from optional libraries.
    }
  });

  child.stderr.on("data", (chunk) => {
    console.error(`[copilot-python] ${String(chunk)}`);
  });

  child.on("exit", () => {
    for (const request of pending.values()) {
      clearTimeout(request.timeout);
      request.reject(new Error("Copilot backend stopped"));
    }
    pending.clear();
    globalThis.copilotBridge = undefined;
  });

  globalThis.copilotBridge = { process: child, pending };
  return globalThis.copilotBridge;
}

function callBackend(payload: unknown) {
  const activeBridge = bridge();
  const id = String(nextId++);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      activeBridge.pending.delete(id);
      reject(new Error("Copilot backend timed out while loading the catalog"));
    }, 25000);
    activeBridge.pending.set(id, { resolve, reject, timeout });
    activeBridge.process.stdin.write(`${JSON.stringify({ id, payload })}\n`);
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const result = await callBackend(body);
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown backend error";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
