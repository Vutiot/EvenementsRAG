import { useState, useRef, useEffect } from "react";

const PREDEFINED_QUESTIONS = [
  "What are the main causes of climate change?",
  "Explain the process of photosynthesis in detail.",
  "What is the difference between SQL and NoSQL databases?",
  "How does the human immune system respond to viruses?",
  "What are the key principles of object-oriented programming?",
  "Describe the architecture of a transformer model.",
  "What are the benefits and risks of nuclear energy?",
  "How does blockchain technology work?",
];

const RAG_PARAMETERS = [
  {
    id: "chunk_size",
    label: "Chunk Size",
    description: "Number of tokens per document chunk",
    type: "discrete",
    options: [128, 256, 512, 1024, 2048],
    defaultIndex: 2,
    unit: "tokens",
    icon: "⊞",
  },
  {
    id: "chunk_overlap",
    label: "Chunk Overlap",
    description: "Overlap between consecutive chunks",
    type: "discrete",
    options: [0, 32, 64, 128, 256],
    defaultIndex: 2,
    unit: "tokens",
    icon: "⊟",
  },
  {
    id: "top_k",
    label: "Top K",
    description: "Number of retrieved documents",
    type: "discrete",
    options: [1, 3, 5, 10, 20],
    defaultIndex: 2,
    unit: "docs",
    icon: "◈",
  },
  {
    id: "embedding_model",
    label: "Embedding Model",
    description: "Model used for vector embeddings",
    type: "categorical",
    options: [
      "text-embedding-3-small",
      "text-embedding-3-large",
      "text-embedding-ada-002",
      "voyage-3",
      "voyage-3-lite",
    ],
    defaultIndex: 0,
    unit: "",
    icon: "◉",
  },
  {
    id: "similarity_metric",
    label: "Similarity Metric",
    description: "Distance function for retrieval",
    type: "categorical",
    options: ["cosine", "euclidean", "dot_product", "manhattan"],
    defaultIndex: 0,
    unit: "",
    icon: "∿",
  },
  {
    id: "temperature",
    label: "Temperature",
    description: "LLM generation temperature",
    type: "continuous",
    min: 0,
    max: 2,
    step: 0.1,
    defaultValue: 0.7,
    unit: "",
    icon: "◐",
  },
  {
    id: "reranker",
    label: "Reranker",
    description: "Reranking model after retrieval",
    type: "categorical",
    options: ["none", "cohere-v3", "bge-reranker-v2", "cross-encoder"],
    defaultIndex: 0,
    unit: "",
    icon: "⇅",
  },
  {
    id: "llm_model",
    label: "LLM Model",
    description: "Language model for generation",
    type: "categorical",
    options: [
      "gpt-4o",
      "gpt-4o-mini",
      "claude-sonnet-4-20250514",
      "claude-haiku",
      "llama-3.1-70b",
    ],
    defaultIndex: 0,
    unit: "",
    icon: "◎",
  },
];

// --- Sub-components ---

function ModeToggle({ mode, setMode }) {
  return (
    <div style={styles.modeToggleContainer}>
      <div style={styles.modeToggleTrack}>
        <div
          style={{
            ...styles.modeToggleSlider,
            transform:
              mode === "question" ? "translateX(0)" : "translateX(100%)",
          }}
        />
        <button
          onClick={() => setMode("question")}
          style={{
            ...styles.modeToggleBtn,
            color: mode === "question" ? "#0a0a0f" : "#8a8a9a",
          }}
        >
          <span style={{ fontSize: 14 }}>?</span> Question
        </button>
        <button
          onClick={() => setMode("evaluation")}
          style={{
            ...styles.modeToggleBtn,
            color: mode === "evaluation" ? "#0a0a0f" : "#8a8a9a",
          }}
        >
          <span style={{ fontSize: 14 }}>◆</span> Evaluation
        </button>
      </div>
    </div>
  );
}

function QuestionPanel({ question, setQuestion }) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target))
        setDropdownOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div style={styles.questionPanel}>
      <div style={styles.questionHeader}>
        <span style={styles.sectionLabel}>QUERY INPUT</span>
      </div>
      <div style={styles.questionBody}>
        <div style={styles.dropdownWrapper} ref={dropdownRef}>
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            style={styles.dropdownTrigger}
          >
            <span style={{ opacity: 0.5, fontSize: 12 }}>▾</span>
            <span>Predefined questions</span>
          </button>
          {dropdownOpen && (
            <div style={styles.dropdownMenu}>
              {PREDEFINED_QUESTIONS.map((q, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setQuestion(q);
                    setDropdownOpen(false);
                  }}
                  style={{
                    ...styles.dropdownItem,
                    background:
                      question === q
                        ? "rgba(209,154,102,0.12)"
                        : "transparent",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background =
                      "rgba(209,154,102,0.08)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background =
                      question === q
                        ? "rgba(209,154,102,0.12)"
                        : "transparent")
                  }
                >
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>
        <div style={styles.chatInputWrapper}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type your question or select one above..."
            style={styles.chatInput}
            rows={3}
          />
          <button
            style={{
              ...styles.sendBtn,
              opacity: question.trim() ? 1 : 0.3,
            }}
            disabled={!question.trim()}
          >
            Run ▸
          </button>
        </div>
      </div>
    </div>
  );
}

function EvaluationBanner() {
  return (
    <div style={styles.evalBanner}>
      <div style={styles.evalBannerIcon}>◆</div>
      <div>
        <div style={styles.evalBannerTitle}>Evaluation Mode</div>
        <div style={styles.evalBannerDesc}>
          Runs the full evaluation dataset against your parameter configuration.
          No single query needed.
        </div>
      </div>
    </div>
  );
}

function ParameterCard({
  param,
  isOptimizing,
  isLocked,
  onToggle,
  value,
  onValueChange,
}) {
  const locked = !isOptimizing && isLocked !== false;
  const optimizing = isOptimizing;

  const getDisplayValue = () => {
    if (param.type === "continuous") {
      return `${value}${param.unit ? " " + param.unit : ""}`;
    }
    const v = param.options[value];
    return `${v}${param.unit ? " " + param.unit : ""}`;
  };

  return (
    <div
      style={{
        ...styles.paramCard,
        borderColor: optimizing
          ? "#d19a66"
          : locked
          ? "rgba(255,255,255,0.04)"
          : "rgba(255,255,255,0.08)",
        background: optimizing
          ? "linear-gradient(135deg, rgba(209,154,102,0.08) 0%, rgba(209,154,102,0.02) 100%)"
          : locked
          ? "rgba(255,255,255,0.01)"
          : "rgba(255,255,255,0.02)",
        opacity: locked ? 0.55 : 1,
      }}
    >
      {/* Card Header */}
      <div style={styles.paramCardHeader}>
        <div style={styles.paramCardLeft}>
          <span
            style={{
              ...styles.paramIcon,
              color: optimizing ? "#d19a66" : "#6a6a7a",
            }}
          >
            {param.icon}
          </span>
          <div>
            <div
              style={{
                ...styles.paramLabel,
                color: optimizing ? "#d19a66" : "#e0e0e8",
              }}
            >
              {param.label}
            </div>
            <div style={styles.paramDesc}>{param.description}</div>
          </div>
        </div>
        <button
          onClick={onToggle}
          style={{
            ...styles.lockBtn,
            background: optimizing
              ? "rgba(209,154,102,0.2)"
              : "rgba(255,255,255,0.05)",
            borderColor: optimizing
              ? "rgba(209,154,102,0.4)"
              : "rgba(255,255,255,0.08)",
            color: optimizing ? "#d19a66" : locked ? "#4a4a5a" : "#8a8a9a",
          }}
          title={optimizing ? "Currently optimizing" : locked ? "Click to optimize this parameter" : "Fixed"}
        >
          {optimizing ? (
            <span style={{ fontSize: 13 }}>⟳ OPTIMIZE</span>
          ) : (
            <span style={{ fontSize: 13 }}>⏚ FIXED</span>
          )}
        </button>
      </div>

      {/* Value Control */}
      <div style={styles.paramCardBody}>
        {optimizing ? (
          <div style={styles.optimizeNotice}>
            <span style={styles.optimizePulse} />
            All values will be swept during optimization
          </div>
        ) : param.type === "continuous" ? (
          <div style={styles.sliderRow}>
            <span style={styles.sliderLabel}>{param.min}</span>
            <input
              type="range"
              min={param.min}
              max={param.max}
              step={param.step}
              value={value}
              onChange={(e) => onValueChange(parseFloat(e.target.value))}
              style={styles.slider}
              disabled={optimizing}
            />
            <span style={styles.sliderValue}>{value}</span>
          </div>
        ) : (
          <div style={styles.optionChips}>
            {param.options.map((opt, idx) => (
              <button
                key={idx}
                onClick={() => onValueChange(idx)}
                disabled={optimizing}
                style={{
                  ...styles.chip,
                  background:
                    value === idx
                      ? "rgba(209,154,102,0.15)"
                      : "rgba(255,255,255,0.04)",
                  borderColor:
                    value === idx
                      ? "rgba(209,154,102,0.5)"
                      : "rgba(255,255,255,0.06)",
                  color: value === idx ? "#d19a66" : "#8a8a9a",
                }}
              >
                {opt}
                {param.unit ? (
                  <span style={{ opacity: 0.5, marginLeft: 2, fontSize: 10 }}>
                    {param.unit}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
        )}

        {!optimizing && (
          <div style={styles.fixedValueTag}>
            Fixed at: <strong>{getDisplayValue()}</strong>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Main App ---

export default function RAGOptimizer() {
  const [mode, setMode] = useState("question");
  const [question, setQuestion] = useState("");
  const [optimizingParam, setOptimizingParam] = useState(null);
  const [paramValues, setParamValues] = useState(() => {
    const vals = {};
    RAG_PARAMETERS.forEach((p) => {
      if (p.type === "continuous") vals[p.id] = p.defaultValue;
      else vals[p.id] = p.defaultIndex;
    });
    return vals;
  });

  const handleToggle = (paramId) => {
    setOptimizingParam((prev) => (prev === paramId ? null : paramId));
  };

  const handleValueChange = (paramId, val) => {
    setParamValues((prev) => ({ ...prev, [paramId]: val }));
  };

  const optimizingMeta = RAG_PARAMETERS.find((p) => p.id === optimizingParam);

  return (
    <div style={styles.root}>
      {/* Bg grain */}
      <div style={styles.grain} />

      <div style={styles.container}>
        {/* Header */}
        <header style={styles.header}>
          <div style={styles.headerLeft}>
            <div style={styles.logo}>
              <span style={styles.logoIcon}>◈</span>
            </div>
            <div>
              <h1 style={styles.title}>RAG Parameter Optimizer</h1>
              <p style={styles.subtitle}>
                Select one parameter to optimize · Fix the rest
              </p>
            </div>
          </div>
          <ModeToggle mode={mode} setMode={setMode} />
        </header>

        {/* Mode-specific top section */}
        {mode === "question" ? (
          <QuestionPanel question={question} setQuestion={setQuestion} />
        ) : (
          <EvaluationBanner />
        )}

        {/* Optimization status strip */}
        <div style={styles.statusStrip}>
          <div style={styles.statusLeft}>
            <span style={styles.statusDot(!!optimizingParam)} />
            {optimizingParam ? (
              <span>
                Optimizing{" "}
                <strong style={{ color: "#d19a66" }}>
                  {optimizingMeta?.label}
                </strong>{" "}
                — {RAG_PARAMETERS.length - 1} parameters fixed
              </span>
            ) : (
              <span style={{ opacity: 0.5 }}>
                Click <em>FIXED</em> on any parameter below to set it as the
                optimization target
              </span>
            )}
          </div>
          {optimizingParam && (
            <button
              style={styles.runOptBtn}
              onClick={() => {}}
            >
              Run Optimization ▸
            </button>
          )}
        </div>

        {/* Parameter Grid */}
        <div style={styles.paramGrid}>
          {RAG_PARAMETERS.map((param) => (
            <ParameterCard
              key={param.id}
              param={param}
              isOptimizing={optimizingParam === param.id}
              isLocked={optimizingParam && optimizingParam !== param.id}
              onToggle={() => handleToggle(param.id)}
              value={paramValues[param.id]}
              onValueChange={(v) => handleValueChange(param.id, v)}
            />
          ))}
        </div>

        {/* Footer */}
        <footer style={styles.footer}>
          <span style={{ opacity: 0.3 }}>
            RAG Optimizer v0.1 — Parameter sweep tool
          </span>
        </footer>
      </div>
    </div>
  );
}

// --- Styles ---
const styles = {
  root: {
    minHeight: "100vh",
    background: "#0a0a0f",
    color: "#c8c8d4",
    fontFamily: "'DM Sans', 'Segoe UI', system-ui, sans-serif",
    position: "relative",
    overflow: "hidden",
  },
  grain: {
    position: "fixed",
    inset: 0,
    opacity: 0.03,
    backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
    pointerEvents: "none",
    zIndex: 0,
  },
  container: {
    position: "relative",
    zIndex: 1,
    maxWidth: 960,
    margin: "0 auto",
    padding: "32px 24px 64px",
  },

  // Header
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 32,
    flexWrap: "wrap",
    gap: 16,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: 14,
  },
  logo: {
    width: 42,
    height: 42,
    borderRadius: 10,
    background: "linear-gradient(135deg, rgba(209,154,102,0.2), rgba(209,154,102,0.05))",
    border: "1px solid rgba(209,154,102,0.25)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  logoIcon: {
    fontSize: 20,
    color: "#d19a66",
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 600,
    color: "#eaeaf2",
    letterSpacing: "-0.02em",
    fontFamily: "'DM Sans', system-ui, sans-serif",
  },
  subtitle: {
    margin: 0,
    fontSize: 13,
    color: "#6a6a7a",
    marginTop: 2,
  },

  // Mode toggle
  modeToggleContainer: {},
  modeToggleTrack: {
    position: "relative",
    display: "flex",
    background: "rgba(255,255,255,0.04)",
    borderRadius: 10,
    border: "1px solid rgba(255,255,255,0.06)",
    overflow: "hidden",
  },
  modeToggleSlider: {
    position: "absolute",
    top: 2,
    left: 2,
    width: "calc(50% - 2px)",
    height: "calc(100% - 4px)",
    borderRadius: 8,
    background: "rgba(255,255,255,0.08)",
    transition: "transform 0.25s cubic-bezier(0.4,0,0.2,1)",
    pointerEvents: "none",
  },
  modeToggleBtn: {
    position: "relative",
    zIndex: 1,
    border: "none",
    background: "none",
    padding: "8px 18px",
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    transition: "color 0.2s",
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontFamily: "inherit",
  },

  // Question panel
  questionPanel: {
    background: "rgba(255,255,255,0.02)",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 14,
    marginBottom: 20,
    overflow: "hidden",
  },
  questionHeader: {
    padding: "12px 18px",
    borderBottom: "1px solid rgba(255,255,255,0.04)",
  },
  sectionLabel: {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: "0.1em",
    color: "#5a5a6a",
    textTransform: "uppercase",
  },
  questionBody: {
    padding: 18,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  dropdownWrapper: {
    position: "relative",
  },
  dropdownTrigger: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 14px",
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 8,
    color: "#9a9aaa",
    fontSize: 13,
    cursor: "pointer",
    fontFamily: "inherit",
    transition: "border-color 0.2s",
    width: "100%",
    textAlign: "left",
  },
  dropdownMenu: {
    position: "absolute",
    top: "calc(100% + 4px)",
    left: 0,
    right: 0,
    background: "#151520",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10,
    padding: 4,
    zIndex: 100,
    maxHeight: 260,
    overflowY: "auto",
    boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
  },
  dropdownItem: {
    display: "block",
    width: "100%",
    textAlign: "left",
    padding: "10px 14px",
    border: "none",
    background: "transparent",
    color: "#b8b8c8",
    fontSize: 13,
    cursor: "pointer",
    borderRadius: 7,
    fontFamily: "inherit",
    lineHeight: 1.4,
    transition: "background 0.15s",
  },
  chatInputWrapper: {
    display: "flex",
    gap: 10,
    alignItems: "flex-end",
  },
  chatInput: {
    flex: 1,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10,
    padding: "12px 16px",
    color: "#e0e0e8",
    fontSize: 14,
    fontFamily: "inherit",
    resize: "none",
    outline: "none",
    lineHeight: 1.5,
    transition: "border-color 0.2s",
  },
  sendBtn: {
    padding: "12px 22px",
    background: "linear-gradient(135deg, #d19a66, #c08050)",
    border: "none",
    borderRadius: 10,
    color: "#0a0a0f",
    fontWeight: 600,
    fontSize: 13,
    cursor: "pointer",
    fontFamily: "inherit",
    whiteSpace: "nowrap",
    transition: "opacity 0.2s, transform 0.15s",
    letterSpacing: "0.02em",
  },

  // Eval banner
  evalBanner: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: "20px 22px",
    background:
      "linear-gradient(135deg, rgba(130,100,220,0.08), rgba(130,100,220,0.02))",
    border: "1px solid rgba(130,100,220,0.15)",
    borderRadius: 14,
    marginBottom: 20,
  },
  evalBannerIcon: {
    fontSize: 24,
    color: "#8264dc",
    width: 44,
    height: 44,
    borderRadius: 10,
    background: "rgba(130,100,220,0.12)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  evalBannerTitle: {
    fontSize: 15,
    fontWeight: 600,
    color: "#c0b4e8",
    marginBottom: 2,
  },
  evalBannerDesc: {
    fontSize: 13,
    color: "#8a80aa",
    lineHeight: 1.4,
  },

  // Status strip
  statusStrip: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 18px",
    background: "rgba(255,255,255,0.02)",
    border: "1px solid rgba(255,255,255,0.04)",
    borderRadius: 10,
    marginBottom: 20,
    fontSize: 13,
    flexWrap: "wrap",
    gap: 12,
  },
  statusLeft: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  statusDot: (active) => ({
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: active ? "#d19a66" : "#3a3a4a",
    boxShadow: active ? "0 0 8px rgba(209,154,102,0.4)" : "none",
    flexShrink: 0,
  }),
  runOptBtn: {
    padding: "8px 20px",
    background: "linear-gradient(135deg, #d19a66, #c08050)",
    border: "none",
    borderRadius: 8,
    color: "#0a0a0f",
    fontWeight: 600,
    fontSize: 13,
    cursor: "pointer",
    fontFamily: "inherit",
    letterSpacing: "0.02em",
  },

  // Param grid
  paramGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(420, 1fr))",
    gap: 12,
  },
  paramCard: {
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 12,
    padding: 18,
    transition: "all 0.3s cubic-bezier(0.4,0,0.2,1)",
  },
  paramCardHeader: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 14,
  },
  paramCardLeft: {
    display: "flex",
    alignItems: "flex-start",
    gap: 10,
  },
  paramIcon: {
    fontSize: 18,
    marginTop: 1,
    flexShrink: 0,
  },
  paramLabel: {
    fontSize: 14,
    fontWeight: 600,
    letterSpacing: "-0.01em",
  },
  paramDesc: {
    fontSize: 11,
    color: "#5a5a6a",
    marginTop: 2,
    lineHeight: 1.3,
  },
  lockBtn: {
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 7,
    padding: "5px 12px",
    cursor: "pointer",
    fontFamily: "'DM Mono', 'SF Mono', monospace",
    fontWeight: 500,
    fontSize: 11,
    letterSpacing: "0.04em",
    transition: "all 0.2s",
    whiteSpace: "nowrap",
    flexShrink: 0,
  },
  paramCardBody: {
    minHeight: 36,
  },
  optimizeNotice: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 12,
    color: "#d19a66",
    opacity: 0.8,
    fontStyle: "italic",
  },
  optimizePulse: {
    display: "inline-block",
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "#d19a66",
    animation: "pulse 1.5s ease-in-out infinite",
  },
  sliderRow: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  sliderLabel: {
    fontSize: 11,
    color: "#5a5a6a",
    minWidth: 16,
    textAlign: "right",
  },
  slider: {
    flex: 1,
    height: 4,
    appearance: "none",
    WebkitAppearance: "none",
    background: "rgba(255,255,255,0.08)",
    borderRadius: 2,
    outline: "none",
    cursor: "pointer",
    accentColor: "#d19a66",
  },
  sliderValue: {
    fontSize: 13,
    fontWeight: 600,
    color: "#d19a66",
    minWidth: 28,
    textAlign: "left",
    fontFamily: "'DM Mono', monospace",
  },
  optionChips: {
    display: "flex",
    flexWrap: "wrap",
    gap: 6,
  },
  chip: {
    padding: "5px 12px",
    borderRadius: 6,
    border: "1px solid rgba(255,255,255,0.06)",
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "'DM Mono', 'SF Mono', monospace",
    transition: "all 0.15s",
    whiteSpace: "nowrap",
  },
  fixedValueTag: {
    marginTop: 10,
    fontSize: 11,
    color: "#5a5a6a",
    fontFamily: "'DM Mono', monospace",
  },

  // Footer
  footer: {
    textAlign: "center",
    marginTop: 48,
    fontSize: 11,
    color: "#4a4a5a",
  },
};
