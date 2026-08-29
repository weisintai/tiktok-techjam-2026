const state = {
  bootstrap: null,
  sessionToken: null,
  starterPrompt: "",
  currentSample: null,
  history: [],
};

const presets = [
  "I want something for rocky outdoor walks in wet weather, preferably lightweight.",
  "Not leather. Blue would be nice, and keep it under $90.",
  "I'm browsing for something versatile for work and weekends.",
  "Nothing flashy, but I still want it comfortable.",
  "Actually switch to casual white sneakers instead.",
];

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setStatus(message) {
  document.getElementById("status").textContent = message;
}

function setError(message = "") {
  const banner = document.getElementById("error-banner");
  if (!message) {
    banner.textContent = "";
    banner.classList.add("hidden");
    return;
  }
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function renderTarget(target) {
  const box = document.getElementById("target-card");
  if (!target) {
    box.className = "target-card empty";
    box.textContent = "Custom prompt mode has no fixed ground-truth item. Pick a benchmark sample to compare against the correct target.";
    return;
  }
  box.className = "target-card";
  box.innerHTML = `
    <div class="target-title">${target.title}</div>
    <div class="target-meta">${target.parent_asin}</div>
    <div class="target-meta">${target.categories.join(" > ")}</div>
    <div class="target-meta">${target.store || "Unknown store"}${target.price ? ` • $${target.price}` : ""}</div>
  `;
}

function renderSampleSummary(sample) {
  const root = document.getElementById("sample-summary");
  const pill = document.getElementById("sample-pill");
  if (!sample) {
    pill.textContent = "Custom";
    root.className = "target-card empty";
    root.textContent = "Custom prompt mode has no benchmark ground truth. Use this mode to freestyle prompts.";
    return;
  }
  pill.textContent = `${sample.scenario_type} • ${sample.difficulty_bucket}`;
  root.className = "target-card";
  root.innerHTML = `
    <div class="target-title">${sample.sample_id}</div>
    <div class="target-meta">${sample.profile_summary || "No profile summary available."}</div>
  `;
}

function renderConversation() {
  const root = document.getElementById("conversation-log");
  if (!state.history.length) {
    root.innerHTML = "<div class='conversation-empty'>Session ready. Your prompts and the bot follow-up will appear here.</div>";
    return;
  }
  root.innerHTML = state.history.map((entry) => `
    <div class="bubble ${entry.role}">
      <div class="bubble-label">${entry.role === "user" ? "Shopper" : "Copilot"}</div>
      <div>${entry.text}</div>
    </div>
  `).join("");
}

function renderWhyBox(payload) {
  const lines = [
    `Mode: ${payload.mode}${payload.dense_loaded ? " (semantic model active)" : ""}`,
    `Turn: ${payload.turn}`,
    `Ask attribute: ${payload.ask_attribute}`,
    `Returned this turn: ${payload.returned_count}`,
    `Candidate pool: ${payload.diagnostics.candidate_count}`,
    `Exact-match ties: ${payload.diagnostics.exact_tie_count}`,
    `Dense enabled: ${payload.diagnostics.dense_enabled === 1 ? "yes" : "no"}`,
  ];
  const top = payload.top_10_preview[0];
  if (top) {
    lines.push("");
    lines.push(`Top result: ${top.title}`);
    for (const reason of top.reasons || []) {
      lines.push(`- ${reason}`);
    }
  }
  document.getElementById("why-box").textContent = lines.join("\n");
}

function renderResultList(elementId, items, targetAsin, returnedAsins = []) {
  const root = document.getElementById(elementId);
  root.innerHTML = "";
  if (!items.length) {
    root.innerHTML = "<li class='result-card'>No products returned for this turn.</li>";
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    const classes = ["result-card"];
    if (item.parent_asin === targetAsin) classes.push("correct");
    if (returnedAsins.includes(item.parent_asin)) classes.push("returned");
    li.className = classes.join(" ");
    const badges = [];
    if (item.parent_asin === targetAsin) badges.push("<span class='badge ok'>Correct target</span>");
    if (returnedAsins.includes(item.parent_asin)) badges.push("<span class='badge'>Shown this turn</span>");
    li.innerHTML = `
      <div class="result-title">#${item.rank} ${item.title}</div>
      <div class="result-meta">${item.parent_asin}</div>
      <div class="result-meta">${item.categories.join(" > ")}</div>
      <div class="result-meta">${item.store || "Unknown store"}${item.price ? ` • $${item.price}` : ""}</div>
      ${(item.reasons || []).length ? `<div class="reason-list">${item.reasons.map((reason) => `<div class="reason-chip">${reason}</div>`).join("")}</div>` : ""}
      <div class="badges">${badges.join("")}</div>
    `;
    root.appendChild(li);
  });
}

function renderPresets() {
  const root = document.getElementById("prompt-presets");
  root.innerHTML = "";
  presets.forEach((prompt) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset";
    button.textContent = prompt;
    button.addEventListener("click", () => {
      document.getElementById("prompt").value = prompt;
      setStatus("Preset prompt copied into the prompt box.");
    });
    root.appendChild(button);
  });
}

function fillSampleSelect(samples) {
  const select = document.getElementById("sample-select");
  const help = document.getElementById("sample-help");
  select.innerHTML = "";
  if (!samples.length) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "No benchmark samples loaded";
    select.appendChild(empty);
    help.textContent = "Benchmark samples could not be loaded. You can still use custom prompt mode.";
    return;
  }
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Custom prompt only";
  select.appendChild(blank);
  const groups = new Map();
  for (const sample of samples) {
    const key = sample.scenario_type;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(sample);
  }
  for (const [scenario, items] of groups.entries()) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = `${scenario} (${items.length})`;
    for (const sample of items) {
      const option = document.createElement("option");
      option.value = sample.sample_id;
      option.textContent = `${sample.sample_id} • ${sample.difficulty_bucket}`;
      optgroup.appendChild(option);
    }
    select.appendChild(optgroup);
  }
  help.textContent = `${samples.length} benchmark samples loaded. Pick one to compare against the correct target, or leave custom mode on.`;
}

async function loadBootstrap() {
  setError("");
  state.bootstrap = await request("/api/bootstrap");
  fillSampleSelect(state.bootstrap.samples);
  renderPresets();
  setStatus(`Loaded ${state.bootstrap.sample_count} public benchmark samples.`);
  if (state.bootstrap.samples.length) {
    document.getElementById("sample-select").value = state.bootstrap.samples[0].sample_id;
  }
  await startSession();
}

async function startSession() {
  setError("");
  const sampleId = document.getElementById("sample-select").value;
  const mode = document.getElementById("mode").value;
  const payload = await request("/api/session", {
    method: "POST",
    body: JSON.stringify({ sample_id: sampleId, mode }),
  });
  state.sessionToken = payload.session_token;
  state.starterPrompt = payload.starter_prompt || "";
  state.currentSample = payload.sample || null;
  state.history = [];
  if (payload.starter_prompt) {
    document.getElementById("prompt").value = payload.starter_prompt;
  } else if (!document.getElementById("prompt").value.trim()) {
    document.getElementById("prompt").value = "";
  }
  renderTarget(payload.sample?.target || null);
  renderSampleSummary(payload.sample || null);
  renderConversation();
  document.getElementById("target-rank-visible").textContent = "-";
  document.getElementById("target-rank-all").textContent = "-";
  document.getElementById("target-returned").textContent = "-";
  document.getElementById("scenario-type").textContent = payload.sample?.scenario_type || "Custom";
  document.getElementById("state-box").textContent = "Session ready. Send a prompt to inspect state.";
  document.getElementById("response-note").textContent = payload.dense_loaded
    ? "Dense semantic model is active for this session."
    : mode === "dense"
      ? "Dense mode selected, but the embedding model is not currently loaded. The agent will fall back."
      : "Baseline retrieval is active for this session.";
  document.getElementById("returned-list").innerHTML = "";
  document.getElementById("preview-list").innerHTML = "";
  document.getElementById("why-box").textContent = "Send a prompt to see the ranking rationale.";
  document.getElementById("returned-count").textContent = "0 items";
  document.getElementById("workflow-tip").textContent = payload.sample
    ? `Loaded ${payload.sample.sample_id}. You can send the starter prompt as-is or rewrite it in your own words.`
    : "Custom prompt mode is ready. Type any shopper message and press Send.";
  setStatus(payload.sample
    ? `Started ${payload.sample.sample_id}. Session created automatically.`
    : "Started a custom free-prompt session.");
}

async function sendPrompt() {
  if (!state.sessionToken) {
    await startSession();
  return;
  }
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) {
    setStatus("Enter a prompt before sending.");
    return;
  }
  setError("");
  setStatus("Running the agent...");
  const payload = await request("/api/respond", {
    method: "POST",
    body: JSON.stringify({ session_token: state.sessionToken, prompt, top_k: 10 }),
  });
  state.history.push({ role: "user", text: prompt });
  state.history.push({ role: "assistant", text: payload.message });
  const returnedAsins = payload.returned.map((item) => item.parent_asin);
  renderTarget(payload.target);
  renderConversation();
  renderResultList("returned-list", payload.returned, payload.target?.parent_asin, returnedAsins);
  renderResultList("preview-list", payload.top_10_preview, payload.target?.parent_asin, returnedAsins);
  renderWhyBox(payload);
  document.getElementById("returned-count").textContent = `${payload.returned_count} item${payload.returned_count === 1 ? "" : "s"}`;
  document.getElementById("target-rank-visible").textContent = payload.target_rank_visible ?? "Not ranked";
  document.getElementById("target-rank-all").textContent = payload.target_rank_all ?? "Not ranked";
  document.getElementById("target-returned").textContent = payload.target_in_returned ? "Yes" : "No";
  document.getElementById("scenario-type").textContent = state.currentSample?.scenario_type || "Custom";
  document.getElementById("state-box").textContent = JSON.stringify(payload.state, null, 2);
  document.getElementById("response-note").textContent =
    `Turn ${payload.turn} • ask_attribute=${payload.ask_attribute} • ${payload.message}`;
  setStatus(payload.dense_loaded
    ? "Response ready. Semantic retrieval is active."
    : "Response ready. If you expected embeddings, install the dense dependencies first.");
  document.getElementById("workflow-tip").textContent =
    "Keep chatting in the same session to test multi-turn accumulation and overrides.";
}

document.getElementById("reset-session").addEventListener("click", () => {
  startSession().catch((error) => {
    setStatus(`Failed to start session: ${error.message}`);
    setError("The demo could not start a session. Refresh the page or restart the local server.");
  });
});

document.getElementById("send-prompt").addEventListener("click", () => {
  sendPrompt().catch((error) => {
    setStatus(`Failed to send prompt: ${error.message}`);
    setError("The prompt request failed. Check that the local demo server is still running.");
  });
});

document.getElementById("use-starter").addEventListener("click", () => {
  document.getElementById("prompt").value = state.starterPrompt || "";
  setStatus(state.starterPrompt ? "Starter prompt copied into the prompt box." : "No sample starter prompt for this session.");
});

document.getElementById("mode").addEventListener("change", () => {
  startSession().catch((error) => {
    setStatus(`Failed to switch mode: ${error.message}`);
    setError("The demo could not switch modes cleanly. Try resetting the session.");
  });
});

document.getElementById("sample-select").addEventListener("change", () => {
  startSession().catch((error) => {
    setStatus(`Failed to load sample: ${error.message}`);
    setError("The selected benchmark sample could not be loaded.");
  });
});

document.getElementById("prompt").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendPrompt().catch((error) => setStatus(`Failed to send prompt: ${error.message}`));
  }
});

loadBootstrap().catch((error) => {
  setStatus(`Failed to load demo data: ${error.message}`);
  setError("Benchmark sample data failed to load. The dropdown will stay empty until the local API is available.");
  fillSampleSelect([]);
  renderPresets();
});
