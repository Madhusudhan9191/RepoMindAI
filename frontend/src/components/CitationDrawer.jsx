import React, { useEffect } from "react";
import Prism from "prismjs";
import { X, Copy, Terminal } from "lucide-react";

export default function CitationDrawer({ citation, onClose, addToast }) {
  useEffect(() => {
    if (citation) {
      Prism.highlightAll();
    }
  }, [citation]);

  if (!citation) return null;

  const handleCopyCode = () => {
    if (citation.text || citation.content) {
      const code = citation.text || citation.content;
      navigator.clipboard.writeText(code);
      addToast("Code snippet copied to clipboard!", "success");
    }
  };

  // The backend citation object could store code text in .text or .content
  const codeContent = citation.text || citation.content || "# No code chunk snippet cached.";
  // Map language to simple highlight format
  const languageClass = citation.file.endsWith(".py") ? "language-python" : "language-javascript";

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer-panel">
        <header className="drawer-header">
          <div className="drawer-header-info">
            <Terminal size={16} style={{ color: "#f59e0b" }} />
            <span className="drawer-title">
              {citation.file} (Lines {citation.start_line}–{citation.end_line})
            </span>
            <span className="drawer-score-badge">
              Similarity: {(citation.score * 100).toFixed(1)}%
            </span>
            <span
              style={{
                background: "rgba(99,102,241,0.18)",
                color: "#a5b4fc",
                padding: "2px 8px",
                borderRadius: "12px",
                fontSize: "11px",
                fontWeight: "600",
                border: "1px solid rgba(99,102,241,0.3)"
              }}
            >
              AST: {citation.type || "Function/Class"}
            </span>
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <button className="btn btn-secondary" style={{ padding: "6px 12px" }} onClick={handleCopyCode}>
              <Copy size={14} /> Copy Code
            </button>
            <button
              onClick={onClose}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                padding: "4px"
              }}
            >
              <X size={18} />
            </button>
          </div>
        </header>
        <div className="drawer-content">
          <pre className={languageClass} style={{ margin: 0, borderRadius: "8px", overflow: "hidden", border: "1px solid var(--border-color)" }}>
            <code className={languageClass}>{codeContent}</code>
          </pre>
        </div>
      </div>
    </>
  );
}
