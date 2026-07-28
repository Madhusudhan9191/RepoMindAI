import React, { useState } from "react";
import { Settings, Database, List, Cpu, FileClock, Trash2, MessageSquare, RotateCcw, Loader2 } from "lucide-react";
import TreeView from "./TreeView";

export default function Sidebar({
  stats,
  settings,
  files,
  onUpdateSettings,
  onRunIndexing,
  onClearIndex,
  isIndexing,
  isClearing,
  addToast,
  onResetConversation,
  completedTurns,
}) {
  // Local state for forms
  const [provider, setProvider] = useState(settings?.provider || "mock");
  const [model, setModel] = useState(settings?.model || "mock-gpt-4o");
  const [apiKey, setApiKey] = useState("");
  const [apiBase, setApiBase] = useState(settings?.api_base || "");
  const [repoPath, setRepoPath] = useState(".");

  const handleSaveSettings = (e) => {
    e.preventDefault();
    if (!model.trim()) {
      addToast("Model name cannot be empty", "error");
      return;
    }
    onUpdateSettings({
      provider,
      model,
      api_key: apiKey || null,
      api_base: apiBase || null
    });
  };

  const handleIndex = (e) => {
    e.preventDefault();
    if (!repoPath.trim()) {
      addToast("Repository path is required", "error");
      return;
    }
    onRunIndexing({ repoPath, clearExisting: true });
  };

  return (
    <aside className="sidebar glass-panel">
      <div className="sidebar-header">
        <Cpu size={24} style={{ color: "#6366f1" }} />
        <h1>RepoMindAI</h1>
      </div>

      <div className="sidebar-scrollable">
        {/* Chat Session — New Chat + turn count */}
        <section className="section-card">
          <h2>
            <MessageSquare size={14} style={{ color: "#6366f1" }} />
            Chat Session
          </h2>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              {completedTurns > 0
                ? <><strong style={{ color: "#fff" }}>{completedTurns}</strong> {completedTurns === 1 ? "exchange" : "exchanges"} in context</>
                : <span style={{ color: "var(--text-muted)" }}>No conversation yet</span>
              }
            </span>
            <button
              id="new-chat-btn"
              type="button"
              className="btn btn-secondary"
              onClick={onResetConversation}
              disabled={completedTurns === 0}
              title="Clear conversation history and start a new chat"
              style={{ padding: "5px 12px", fontSize: "0.78rem", gap: "6px" }}
            >
              <RotateCcw size={12} />
              New Chat
            </button>
          </div>
        </section>

        {/* Model Configuration */}
        <section className="section-card">
          <h2>
            <Settings size={14} style={{ color: "#818cf8" }} />
            Models & Provider
          </h2>
          <form onSubmit={handleSaveSettings} style={{ display: "flex", flex: 1, flexDirection: "column", gap: "10px" }}>
            <div className="input-group">
              <label className="input-label">Provider</label>
              <select
                className="input-field"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
              >
                <option value="mock">Mock LLM</option>
                <option value="ollama">Ollama (Local)</option>
                <option value="openai">OpenAI (GPT)</option>
                <option value="gemini">Gemini (Google)</option>
              </select>
            </div>

            <div className="input-group">
              <label className="input-label">Model Identifier</label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. gpt-4o, llama3"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
            </div>

            {(provider === "openai" || provider === "gemini") && (
              <div className="input-group">
                <label className="input-label">API Key</label>
                <input
                  type="password"
                  className="input-field"
                  placeholder="Enter key (optional if set in env)"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
              </div>
            )}

            {provider === "ollama" && (
              <div className="input-group">
                <label className="input-label">Ollama API Base URL</label>
                <input
                  type="text"
                  className="input-field"
                  placeholder="http://localhost:11434"
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                />
              </div>
            )}

            <button type="submit" className="btn btn-secondary" style={{ marginTop: "4px" }}>
              Apply Changes
            </button>
          </form>
        </section>

        {/* Index Ingestion Trigger */}
        <section className="section-card">
          <h2>
            <Database size={14} style={{ color: "#10b981" }} />
            Repository Index
          </h2>
          <form onSubmit={handleIndex} style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div className="input-group">
              <label className="input-label">Local Repo Path</label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. ."
                value={repoPath}
                onChange={(e) => setRepoPath(e.target.value)}
              />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginTop: "4px" }}>
              <button type="submit" className="btn" disabled={isIndexing}>
                {isIndexing ? (
                  <>
                    <Loader2 size={14} className="animate-spin" /> Indexing...
                  </>
                ) : (
                  "Index Repo"
                )}
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={onClearIndex}
                disabled={isClearing}
                title="Wipe database vectors"
              >
                {isClearing ? (
                  <>
                    <Loader2 size={14} className="animate-spin" /> Clearing...
                  </>
                ) : (
                  <>
                    <Trash2 size={14} /> Clear Index
                  </>
                )}
              </button>
            </div>
          </form>
        </section>

        {/* Statistics Dashboard */}
        <section className="section-card">
          <h2>
            <FileClock size={14} style={{ color: "#f59e0b" }} />
            Database Stats
          </h2>
          <div style={{ fontSize: "0.82rem", display: "grid", gap: "8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Status:</span>
              <span style={{ color: stats?.status === "healthy" ? "var(--success)" : "var(--danger)", fontWeight: 600 }}>
                {stats?.status ? stats.status.toUpperCase() : "DISCONNECTED"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Vectors Count:</span>
              <span style={{ color: "#fff", fontWeight: 600 }}>{stats?.vector_count ?? 0}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Embedding Model:</span>
              <span style={{ color: "#fff", maxWidth: "160px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }} title={stats?.embedding_model}>
                {stats?.embedding_model?.split("/")?.pop() ?? "N/A"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Dimension:</span>
              <span style={{ color: "#fff" }}>{stats?.dimension ?? "N/A"}</span>
            </div>
          </div>
        </section>

        {/* Directory File Tree Explorer */}
        <section className="section-card" style={{ flex: 1, minHeight: "220px", display: "flex", flexDirection: "column" }}>
          <h2>
            <List size={14} style={{ color: "#a5b4fc" }} />
            Repository Explorer
          </h2>
          <TreeView files={files} />
        </section>
      </div>
    </aside>
  );
}
