import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import Prism from "prismjs";
import { Send, Square, Zap, Search, Brain, FileText, ArrowRight, ThumbsUp, ThumbsDown } from "lucide-react";

// Import Prism styles and language components
import "prismjs/themes/prism-tomorrow.css";
import "prismjs/components/prism-python";
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-typescript";
import "prismjs/components/prism-json";
import "prismjs/components/prism-markdown";

// Streaming state constants — mirror the values from App.jsx STREAM_STATUS
const STARTING   = "starting";
const STREAMING  = "streaming";

export default function ChatArea({
  messages,
  onAskQuestion,
  onStopStreaming,
  streamStatus,
  activeModel,
  onCitationClick,
  onSendFeedback,
}) {
  const [question, setQuestion] = useState("");
  const messagesEndRef = useRef(null);
  const rafRef = useRef(null); // requestAnimationFrame handle for scroll throttle

  // Derive boolean convenience flags from the state machine
  const isActive    = streamStatus === STARTING || streamStatus === STREAMING;
  const isStreaming  = streamStatus === STREAMING;
  const isStarting   = streamStatus === STARTING;

  // Re-run Prism syntax highlighting after messages update
  useEffect(() => {
    Prism.highlightAll();
  }, [messages]);

  // rAF-gated auto-scroll — coalesces multiple state updates into one DOM paint
  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    });
    return () => cancelAnimationFrame(rafRef.current);
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    onAskQuestion(question.trim());
    setQuestion("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-panel glass-panel">
      {/* Chat header */}
      <header className="chat-header">
        <div className="chat-header-title">
          <div className={`status-dot ${isActive ? "loading" : ""}`} />
          <h2 style={{ fontSize: "0.95rem", fontWeight: 600 }}>Active Chat Session</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {/* Stop generating button — visible only when tokens are flowing */}
          {isStreaming && (
            <button
              id="stop-streaming-btn"
              onClick={onStopStreaming}
              className="stop-btn"
              title="Stop generating"
            >
              <Square size={12} fill="currentColor" />
              Stop
            </button>
          )}
          {activeModel && (
            <span style={{ fontSize: "0.75rem", background: "rgba(255,255,255,0.05)", padding: "4px 10px", borderRadius: "100px", color: "var(--text-secondary)", border: "1px solid var(--border-color)" }}>
              Model: <strong style={{ color: "#fff" }}>{activeModel}</strong>
            </span>
          )}
        </div>
      </header>

      {/* Messages stream */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <Zap size={40} style={{ color: "#6366f1", marginBottom: "8px" }} />
            <h3>Ask RepoMindAI</h3>
            <p>
              Query your indexed codebase. Get precise, semantic responses backed by line-by-line source citations and stage latency statistics.
            </p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="message-bubble">
                {msg.role === "user" ? (
                  <p style={{ whiteSpace: "pre-wrap" }}>{msg.content}</p>
                ) : (
                  <div className="markdown-content">
                    <ReactMarkdown
                      components={{
                        code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || "");
                          return !inline && match ? (
                            <pre className={className}>
                              <code className={className} {...props}>
                                {children}
                              </code>
                            </pre>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                    {/* Blinking cursor — only while this message is actively streaming */}
                    {msg.streaming && isStreaming && (
                      <span
                        style={{
                          display: "inline-block",
                          width: "2px",
                          height: "1em",
                          background: "#6366f1",
                          marginLeft: "2px",
                          verticalAlign: "text-bottom",
                          animation: "blink 0.8s step-end infinite",
                        }}
                      />
                    )}
                  </div>
                )}
              </div>

              {/* Citations list for assistant responses */}
              {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                <div className="citations-row">
                  {msg.citations.map((cite, idx) => (
                    <div
                      key={idx}
                      className="citation-pill"
                      onClick={() => onCitationClick(cite)}
                      title={`Similarity score: ${(cite.score * 100).toFixed(1)}%`}
                    >
                      <FileText size={12} />
                      <span>
                        [{idx + 1}] {cite.file.split("/").pop()}:{cite.start_line}-{cite.end_line}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Query reformulation indicator */}
              {msg.role === "assistant" && msg.rewrittenQuery && (
                <div className="search-rewritten-query" style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.75rem", color: "var(--text-muted)", margin: "4px 0 8px 12px" }}>
                  <Search size={11} style={{ opacity: 0.6 }} />
                  <span>Searched for: <strong style={{ color: "var(--text-secondary)", fontStyle: "italic" }}>"{msg.rewrittenQuery}"</strong></span>
                </div>
              )}


              {/* Timing metrics */}
              {msg.role === "assistant" && msg.metrics && (
                <div className="metrics-row">
                  <span className="metric-badge total">
                    <Zap size={10} /> Total: {msg.metrics.total_ms}ms
                  </span>
                  <span className="metric-badge">
                    <Search size={10} /> Retrieval: {msg.metrics.retrieval_ms}ms
                  </span>
                  <span className="metric-badge">
                    <Brain size={10} /> LLM: {msg.metrics.llm_ms}ms
                  </span>
                  {msg.metrics.tokens_per_second > 0 && (
                    <span className="metric-badge">
                      <ArrowRight size={10} /> {msg.metrics.tokens_per_second} tok/s
                    </span>
                  )}
                  {msg.requestId && !msg.streaming && onSendFeedback && (
                    <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "10px" }}>
                      {msg.feedbackRating ? (
                        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontStyle: "italic", display: "flex", alignItems: "center", gap: "2px" }}>
                          ✓ Feedback received
                        </span>
                      ) : (
                        <>
                          <button
                            type="button"
                            onClick={() => onSendFeedback(msg.requestId, "thumbs_up")}
                            style={{ background: "none", border: "none", padding: "2px", color: "var(--text-muted)", cursor: "pointer", display: "flex", alignItems: "center", transition: "color 0.2s" }}
                            title="Helpful"
                            onMouseEnter={(e) => e.currentTarget.style.color = "var(--text-secondary)"}
                            onMouseLeave={(e) => e.currentTarget.style.color = "var(--text-muted)"}
                          >
                            <ThumbsUp size={11} />
                          </button>
                          <button
                            type="button"
                            onClick={() => onSendFeedback(msg.requestId, "thumbs_down")}
                            style={{ background: "none", border: "none", padding: "2px", color: "var(--text-muted)", cursor: "pointer", display: "flex", alignItems: "center", transition: "color 0.2s" }}
                            title="Not helpful"
                            onMouseEnter={(e) => e.currentTarget.style.color = "var(--text-secondary)"}
                            onMouseLeave={(e) => e.currentTarget.style.color = "var(--text-muted)"}
                          >
                            <ThumbsDown size={11} />
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {/* Retrieval loading indicator — shown only in the "starting" phase before tokens arrive */}
        {isStarting && (
          <div className="message assistant">
            <div className="message-bubble" style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-secondary)", fontStyle: "italic", fontSize: "0.85rem" }}>
              <span className="status-dot loading" style={{ width: "6px", height: "6px" }} />
              RepoMindAI is understanding repository context and generating response...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input container */}
      <footer className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-wrapper">
          <textarea
            id="chat-input"
            className="chat-input"
            rows="1"
            placeholder="Ask a question about this repository..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isActive}
          />
          <button
            id="chat-send-btn"
            type="submit"
            className={`chat-input-btn ${question.trim() && !isActive ? "active" : ""}`}
            disabled={!question.trim() || isActive}
          >
            <Send size={16} />
          </button>
        </form>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "8px", fontSize: "0.7rem", color: "var(--text-muted)", padding: "0 4px" }}>
          <span>Press Enter to send, Shift+Enter for new line</span>
          <span>Version 1.0.0</span>
        </div>
      </footer>
    </div>
  );
}
