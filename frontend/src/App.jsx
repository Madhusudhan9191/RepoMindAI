import React, { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import CitationDrawer from "./components/CitationDrawer";
import { getStats } from "./api/stats";
import { getSettings, updateSettings, getFiles } from "./api/settings";
import { runIndexing, clearIndex } from "./api/index";
import { askStream, sendFeedback } from "./api/chat";
import { CheckCircle, AlertTriangle, Info } from "lucide-react";

// ---------------------------------------------------------------------------
// Conversational Memory — Token Budget Constants
// ---------------------------------------------------------------------------
// MAX_HISTORY_MESSAGES: hard cap on total history messages sent per request.
// Must be even (pairs of user + assistant). 10 = 5 complete exchanges.
// MAX_HISTORY_TOKENS: reserved for Phase 11 summarisation — set to Infinity
// so the API shape exists without blocking Phase 10 from shipping.
const MAX_HISTORY_MESSAGES = 10;
const MAX_HISTORY_TOKENS = Infinity; // eslint-disable-line no-unused-vars

/**
 * Builds a bounded, balanced conversation history from the messages array.
 *
 * Rules enforced:
 *   - Only completed (non-streaming) user + assistant messages included.
 *   - Messages are grouped into full exchange pairs [user, assistant].
 *   - Pairs taken from the end (recency-biased), then reversed for chronological order.
 *   - Result always begins with a user turn — structurally balanced.
 *
 * @param {Array}  msgs     - Full messages[] state array.
 * @param {number} maxMsgs  - Maximum total messages to include.
 * @returns {Array} [{role, content}, ...]
 */
function buildHistory(msgs, maxMsgs = MAX_HISTORY_MESSAGES) {
  // Only include completed, non-empty, non-streaming messages
  const completed = msgs.filter(
    (m) =>
      (m.role === "user" || m.role === "assistant") &&
      !m.streaming &&
      m.content?.trim()
  );

  // Build pairs working backwards from the end — ensures recency
  const pairs = [];
  let i = completed.length - 1;
  while (i >= 1 && pairs.length * 2 < maxMsgs) {
    if (
      completed[i].role === "assistant" &&
      completed[i - 1].role === "user"
    ) {
      pairs.push([completed[i - 1], completed[i]]);
      i -= 2;
    } else {
      i -= 1;
    }
  }

  // Reverse to chronological order (oldest → newest)
  pairs.reverse();

  // Flatten to [{role, content}] — only the fields the backend expects
  return pairs.flatMap(([user, asst]) => [
    { role: "user",      content: user.content },
    { role: "assistant", content: asst.content },
  ]);
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
// Valid states and their allowed transitions:
//
//   idle → starting       (user submits a prompt)
//   starting → streaming  (metadata frame received)
//   streaming → completed (done frame received)
//   streaming → cancelled (abort() called — new prompt or unmount)
//   streaming → error     (error frame or network failure)
//   completed → idle      (cleanup in finally)
//   cancelled → idle      (cleanup in finally)
//   error → idle          (cleanup in finally)
//
const STREAM_STATUS = {
  IDLE: "idle",
  STARTING: "starting",
  STREAMING: "streaming",
  COMPLETED: "completed",
  CANCELLED: "cancelled",
  ERROR: "error",
};

export default function App() {
  const queryClient = useQueryClient();

  // App-level state
  const [messages, setMessages] = useState([]);
  const [activeCitation, setActiveCitation] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [repoPath, setRepoPath] = useState(".");

  // Streaming state machine
  const [streamStatus, setStreamStatus] = useState(STREAM_STATUS.IDLE);

  // Refs — survive re-renders without triggering them
  const abortControllerRef = useRef(null); // Current AbortController
  const tokenBufferRef = useRef("");       // Batched token accumulator

  // Abort any active stream when the component unmounts
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  // Custom Toast handler
  const addToast = (message, type = "info") => {
    const id = Date.now() + Math.random().toString(36).substr(2, 9);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  // 1. Fetch Stats (Poll every 4 seconds)
  const { data: stats } = useQuery({
    queryKey: ["stats"],
    queryFn: getStats,
    refetchInterval: 4000,
  });

  // 2. Fetch Active Settings
  const { data: settings } = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
  });

  // 3. Fetch Repository File Tree
  const { data: files } = useQuery({
    queryKey: ["files", repoPath],
    queryFn: () => getFiles(repoPath),
    keepPreviousData: true,
  });

  // 4. Mutation: Update Settings
  const updateSettingsMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: (data) => {
      queryClient.invalidateQueries(["settings"]);
      queryClient.invalidateQueries(["stats"]);
      addToast(`Configuration switched to ${data.provider} (${data.model})!`, "success");
    },
    onError: (err) => {
      addToast(err.message || "Failed to update LLM configuration", "error");
    },
  });

  // 5. Mutation: Ingestion Pipeline
  const runIndexingMutation = useMutation({
    mutationFn: ({ repoPath, clearExisting }) => runIndexing(repoPath, clearExisting),
    onSuccess: (data) => {
      queryClient.invalidateQueries(["stats"]);
      queryClient.invalidateQueries(["files"]);
      addToast(`Repository indexed! Scanned ${data.files_scanned} files, created ${data.chunks} chunks.`, "success");
    },
    onError: (err) => {
      addToast(err.message || "Ingestion pipeline failed", "error");
    },
  });

  // 6. Mutation: Clear vector index
  const clearIndexMutation = useMutation({
    mutationFn: clearIndex,
    onSuccess: () => {
      queryClient.invalidateQueries(["stats"]);
      addToast("Vector database collection cleared successfully.", "info");
    },
    onError: (err) => {
      addToast(err.message || "Failed to clear vector store", "error");
    },
  });

  // ---------------------------------------------------------------------------
  // Streaming handler
  // ---------------------------------------------------------------------------
  const handleAskQuestion = async (question) => {
    // Abort any in-flight stream before starting a new one.
    // This cancels the previous HTTP connection immediately.
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Clear any leftover partial streaming messages from a cancelled stream
    setMessages((prev) => prev.filter((m) => !m.streaming));

    // Append the user message immediately
    setMessages((prev) => [...prev, { role: "user", content: question }]);

    // Insert a blank assistant message slot that will fill progressively
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", citations: [], metrics: null, streaming: true },
    ]);

    // Transition: idle → starting
    setStreamStatus(STREAM_STATUS.STARTING);

    // Token batch buffer — reset before each stream
    tokenBufferRef.current = "";

    // Flush accumulated tokens into React state every 30 ms.
    // This reduces re-renders from ~20+/sec to ≤33/sec for fast models.
    const flushInterval = setInterval(() => {
      if (tokenBufferRef.current) {
        const chunk = tokenBufferRef.current;
        tokenBufferRef.current = "";
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.content = (last.content || "") + chunk;
          updated[updated.length - 1] = last;
          return updated;
        });
      }
    }, 30);

    try {
      for await (const frame of askStream(question, 5, controller.signal, buildHistory(messages))) {
        if (frame.event === "metadata") {
          // Transition: starting → streaming (first meaningful frame arrived)
          setStreamStatus(STREAM_STATUS.STREAMING);

          // Apply citations and rewritten query details immediately
          setMessages((prev) => {
            const updated = [...prev];
            const last = { ...updated[updated.length - 1] };
            last.citations = frame.data.citations ?? [];
            last.model = frame.data.model;
            last.rewrittenQuery = frame.data.rewritten_query ?? null;
            last.requestId = frame.data.request_id ?? null;
            updated[updated.length - 1] = last;
            return updated;
          });

        } else if (frame.event === "token") {
          // Buffer token — the setInterval above will flush to state
          tokenBufferRef.current += frame.data.text ?? "";

        } else if (frame.event === "metrics") {
          setMessages((prev) => {
            const updated = [...prev];
            const last = { ...updated[updated.length - 1] };
            last.metrics = frame.data;
            updated[updated.length - 1] = last;
            return updated;
          });

        } else if (frame.event === "done") {
          // Transition: streaming → completed
          setStreamStatus(STREAM_STATUS.COMPLETED);
          setMessages((prev) => {
            const updated = [...prev];
            const last = { ...updated[updated.length - 1] };
            last.streaming = false;
            updated[updated.length - 1] = last;
            return updated;
          });
          break;

        } else if (frame.event === "error") {
          // Transition: streaming → error
          setStreamStatus(STREAM_STATUS.ERROR);
          setMessages((prev) => prev.filter((m) => !m.streaming));
          addToast(frame.data.message || "AI response failed", "error");
          break;
        }
        // keepalive frames are silently ignored
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // Transition: streaming → cancelled (user submitted new prompt or stopped)
        // Silently clean up — no error toast, the new prompt will handle UI.
        setStreamStatus(STREAM_STATUS.CANCELLED);
        setMessages((prev) => {
          // Only remove the cancelled streaming slot (not the new user message)
          const updated = [...prev];
          const cancelledIdx = updated.findLastIndex((m) => m.streaming);
          if (cancelledIdx !== -1) updated.splice(cancelledIdx, 1);
          return updated;
        });
      } else {
        // Transition: streaming → error (network failure etc.)
        setStreamStatus(STREAM_STATUS.ERROR);
        setMessages((prev) => prev.filter((m) => !m.streaming));
        addToast(err.message || "Failed to connect to streaming endpoint", "error");
      }
    } finally {
      // Drain any remaining buffered tokens before clearing the interval
      clearInterval(flushInterval);
      if (tokenBufferRef.current) {
        const remaining = tokenBufferRef.current;
        tokenBufferRef.current = "";
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.content = (last.content || "") + remaining;
          updated[updated.length - 1] = last;
          return updated;
        });
      }
      // Transition: any terminal state → idle (ready for next prompt)
      setStreamStatus(STREAM_STATUS.IDLE);
    }
  };

  // Stop button handler — aborts the current stream gracefully
  const handleStopStreaming = () => {
    abortControllerRef.current?.abort();
  };

  // Reset conversation — clears all messages and aborts any active stream
  const handleResetConversation = () => {
    abortControllerRef.current?.abort();
    setMessages([]);
    setStreamStatus(STREAM_STATUS.IDLE);
  };

  const handleSendFeedback = async (requestId, rating) => {
    try {
      await sendFeedback(requestId, rating);
      addToast("Feedback submitted, thank you!", "success");
      setMessages((prev) =>
        prev.map((msg) =>
          msg.requestId === requestId ? { ...msg, feedbackRating: rating } : msg
        )
      );
    } catch (err) {
      addToast("Failed to submit feedback: " + err.message, "error");
    }
  };

  // Count of completed exchange pairs for the turn badge
  const completedTurns = Math.floor(
    messages.filter((m) => m.role === "assistant" && !m.streaming && m.content?.trim()).length
  );

  const handleIndexingTrigger = ({ repoPath: path, clearExisting }) => {
    setRepoPath(path);
    runIndexingMutation.mutate({ repoPath: path, clearExisting });
  };

  return (
    <div className="app-container">
      {/* Toast Notifications */}
      <div className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            {toast.type === "success" && <CheckCircle size={16} />}
            {toast.type === "error" && <AlertTriangle size={16} />}
            {toast.type === "info" && <Info size={16} />}
            <span>{toast.message}</span>
          </div>
        ))}
      </div>

      {/* Sidebar navigation */}
      <Sidebar
        stats={stats}
        settings={settings}
        files={files}
        onUpdateSettings={updateSettingsMutation.mutate}
        onRunIndexing={handleIndexingTrigger}
        onClearIndex={clearIndexMutation.mutate}
        isIndexing={runIndexingMutation.isPending || runIndexingMutation.isLoading}
        isClearing={clearIndexMutation.isPending || clearIndexMutation.isLoading}
        addToast={addToast}
        onResetConversation={handleResetConversation}
        completedTurns={completedTurns}
      />

      {/* Core chat window */}
      <ChatArea
        messages={messages}
        onAskQuestion={handleAskQuestion}
        onStopStreaming={handleStopStreaming}
        streamStatus={streamStatus}
        activeModel={stats?.llm_model}
        onCitationClick={setActiveCitation}
        onSendFeedback={handleSendFeedback}
      />

      {/* Slide-up Citation Drawer */}
      <CitationDrawer
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
        addToast={addToast}
      />
    </div>
  );
}
