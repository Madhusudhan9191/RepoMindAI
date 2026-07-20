import { request } from "./client";

const BASE_URL = "http://localhost:8000/api/v1";

export function ask(question, topK = 5) {
  return request("/ask", {
    method: "POST",
    body: JSON.stringify({ question, top_k: topK }),
  });
}

/**
 * Streaming RAG query via SSE.
 *
 * Returns an async generator that yields parsed SSE event objects:
 *   { event: "metadata" | "token" | "metrics" | "done" | "error" | "keepalive", data: {} }
 *
 * @param {string}      question  - The user's question.
 * @param {number}      topK      - Number of chunks to retrieve (default 5).
 * @param {AbortSignal} signal    - Optional AbortSignal to cancel the fetch.
 * @param {Array}       history   - Prior conversation turns [{role, content}].
 *
 * Usage:
 *   const ac = new AbortController();
 *   for await (const frame of askStream(q, 5, ac.signal, history)) { ... }
 *   ac.abort(); // cancels the in-flight HTTP connection
 */
export async function* askStream(question, topK = 5, signal = null, history = []) {
  const fetchOptions = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, history }),
  };
  if (signal) fetchOptions.signal = signal;

  const response = await fetch(`${BASE_URL}/ask/stream`, fetchOptions);


  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by double newlines
    const frames = buffer.split("\n\n");
    // Keep the last incomplete frame in the buffer
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      if (!frame.trim()) continue;

      let eventType = "message";
      let dataLine = "";

      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLine = line.slice(5).trim();
        }
      }

      let parsedData = {};
      try {
        parsedData = dataLine ? JSON.parse(dataLine) : {};
      } catch {
        parsedData = { raw: dataLine };
      }

      yield { event: eventType, data: parsedData };

      // Stop iteration after terminal events
      if (eventType === "done" || eventType === "error") {
        return;
      }
    }
  }
}

export async function sendFeedback(requestId, rating, comment = null) {
  const response = await fetch(`${BASE_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, rating, feedback_text: comment }),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP ${response.status}`);
  }
  return response.json();
}
