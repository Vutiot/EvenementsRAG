import { useState, useMemo, useRef, useCallback } from "react";
import _ from "lodash";

const COLORS = [
  "#818cf8","#22d3ee","#facc15","#f87171","#34d399","#f472b6",
  "#a78bfa","#2dd4bf","#fb923c","#94a3b8","#c084fc","#38bdf8",
  "#fbbf24","#4ade80","#e879f9","#67e8f9","#fca5a1","#86efac"
];

const PARAM_KEYS = ["chunk_size","overlap","chunk_method","embedding","context_embedding","retrieval","reranker","hybrid_method","top_k"];
const METRIC_KEYS = ["hit_rate","mrr","ndcg_10","recall_20","faithfulness","latency_ms","cost_per_1k"];
const PARAM_LABELS = {chunk_size:"Chunk",overlap:"Overlap",chunk_method:"Method",embedding:"Embed",context_embedding:"Ctx embed",retrieval:"Retrieval",reranker:"Reranker",hybrid_method:"Hybrid",top_k:"Top-K"};
const METRIC_LABELS = {hit_rate:"Hit rate",mrr:"MRR",ndcg_10:"NDCG@10",recall_20:"Recall@20",faithfulness:"Faith.",latency_ms:"Latency",cost_per_1k:"Cost"};
const HIGHER_BETTER = {hit_rate:true,mrr:true,ndcg_10:true,recall_20:true,faithfulness:true,latency_ms:false,cost_per_1k:false};

const SAMPLE_DATA = [
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"OpenAI-3-sm",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:5,hit_rate:0.72,mrr:0.65,ndcg_10:0.58,recall_20:0.78,faithfulness:0.71,latency_ms:210,cost_per_1k:0.12},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"OpenAI-3-sm",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:5,hit_rate:0.81,mrr:0.74,ndcg_10:0.67,recall_20:0.85,faithfulness:0.78,latency_ms:380,cost_per_1k:0.18},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"OpenAI-3-lg",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:5,hit_rate:0.87,mrr:0.81,ndcg_10:0.74,recall_20:0.91,faithfulness:0.83,latency_ms:460,cost_per_1k:0.22},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"VoyageRR-2.5",hybrid_method:"none",top_k:5,hit_rate:0.89,mrr:0.83,ndcg_10:0.76,recall_20:0.93,faithfulness:0.85,latency_ms:390,cost_per_1k:0.21},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-sm",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.76,mrr:0.68,ndcg_10:0.62,recall_20:0.82,faithfulness:0.74,latency_ms:230,cost_per_1k:0.13},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-sm",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.85,mrr:0.78,ndcg_10:0.71,recall_20:0.89,faithfulness:0.81,latency_ms:410,cost_per_1k:0.19},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-lg",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.90,mrr:0.84,ndcg_10:0.77,recall_20:0.94,faithfulness:0.86,latency_ms:490,cost_per_1k:0.23},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-lg",context_embedding:"none",retrieval:"dense",reranker:"bge-rr-lg",hybrid_method:"none",top_k:10,hit_rate:0.88,mrr:0.82,ndcg_10:0.75,recall_20:0.92,faithfulness:0.84,latency_ms:350,cost_per_1k:0.17},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"VoyageRR-2.5",hybrid_method:"none",top_k:10,hit_rate:0.92,mrr:0.86,ndcg_10:0.79,recall_20:0.95,faithfulness:0.88,latency_ms:410,cost_per_1k:0.22},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"JinaAI-v2",context_embedding:"none",retrieval:"dense",reranker:"bge-rr-lg",hybrid_method:"none",top_k:10,hit_rate:0.93,mrr:0.87,ndcg_10:0.81,recall_20:0.96,faithfulness:0.89,latency_ms:340,cost_per_1k:0.16},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-lg",context_embedding:"contextual",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.88,mrr:0.82,ndcg_10:0.75,recall_20:0.92,faithfulness:0.84,latency_ms:320,cost_per_1k:0.28},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-lg",context_embedding:"contextual",retrieval:"hybrid",reranker:"none",hybrid_method:"RRF-k60",top_k:10,hit_rate:0.91,mrr:0.85,ndcg_10:0.78,recall_20:0.95,faithfulness:0.87,latency_ms:380,cost_per_1k:0.30},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-lg",context_embedding:"contextual",retrieval:"hybrid",reranker:"CohereRerank",hybrid_method:"RRF-k60",top_k:10,hit_rate:0.95,mrr:0.91,ndcg_10:0.85,recall_20:0.98,faithfulness:0.92,latency_ms:520,cost_per_1k:0.36},
  {chunk_size:512,overlap:10,chunk_method:"semantic",embedding:"OpenAI-3-lg",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.91,mrr:0.85,ndcg_10:0.78,recall_20:0.94,faithfulness:0.87,latency_ms:510,cost_per_1k:0.25},
  {chunk_size:512,overlap:10,chunk_method:"late_chunk",embedding:"JinaAI-v2",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.89,mrr:0.83,ndcg_10:0.77,recall_20:0.93,faithfulness:0.85,latency_ms:200,cost_per_1k:0.12},
  {chunk_size:512,overlap:10,chunk_method:"late_chunk",embedding:"JinaAI-v2",context_embedding:"none",retrieval:"dense",reranker:"bge-rr-lg",hybrid_method:"none",top_k:10,hit_rate:0.94,mrr:0.89,ndcg_10:0.83,recall_20:0.97,faithfulness:0.91,latency_ms:360,cost_per_1k:0.17},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"BGE-M3",context_embedding:"none",retrieval:"hybrid",reranker:"bge-rr-lg",hybrid_method:"RRF-k60",top_k:10,hit_rate:0.92,mrr:0.86,ndcg_10:0.80,recall_20:0.96,faithfulness:0.89,latency_ms:400,cost_per_1k:0.15},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"BGE-M3",context_embedding:"none",retrieval:"hybrid",reranker:"none",hybrid_method:"SPLADE+dns",top_k:10,hit_rate:0.89,mrr:0.83,ndcg_10:0.76,recall_20:0.93,faithfulness:0.85,latency_ms:280,cost_per_1k:0.11},
  {chunk_size:1024,overlap:20,chunk_method:"recursive",embedding:"OpenAI-3-lg",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.88,mrr:0.82,ndcg_10:0.75,recall_20:0.92,faithfulness:0.85,latency_ms:510,cost_per_1k:0.24},
  {chunk_size:1024,overlap:20,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"VoyageRR-2.5",hybrid_method:"none",top_k:10,hit_rate:0.90,mrr:0.84,ndcg_10:0.77,recall_20:0.94,faithfulness:0.87,latency_ms:430,cost_per_1k:0.23},
  {chunk_size:1024,overlap:20,chunk_method:"page_level",embedding:"OpenAI-3-lg",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.87,mrr:0.81,ndcg_10:0.74,recall_20:0.91,faithfulness:0.84,latency_ms:480,cost_per_1k:0.23},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"contextual",retrieval:"hybrid",reranker:"VoyageRR-2.5",hybrid_method:"RRF-k60",top_k:20,hit_rate:0.96,mrr:0.92,ndcg_10:0.87,recall_20:0.98,faithfulness:0.93,latency_ms:580,cost_per_1k:0.42},
];

function getUniqueVals(data, key) {
  return [...new Set(data.map(d => d[key]))].sort((a,b) => typeof a === "number" ? a-b : String(a).localeCompare(String(b)));
}

function buildConfigs(data) {
  const grouped = _.groupBy(data, d => PARAM_KEYS.map(k => d[k]).join("|"));
  return Object.entries(grouped).map(([key, rows]) => {
    const params = {};
    PARAM_KEYS.forEach(k => { params[k] = rows[0][k]; });
    const metrics = {};
    METRIC_KEYS.forEach(k => {
      metrics[k] = _.mean(rows.map(r => r[k]));
    });
    return { ...params, ...metrics, _count: rows.length };
  });
}

function smartLabel(configs) {
  const varyingKeys = PARAM_KEYS.filter(k => {
    const vals = new Set(configs.map(c => String(c[k])));
    return vals.size > 1;
  });
  if (varyingKeys.length === 0) return configs.map((_, i) => `Config ${i+1}`);
  const maxKeys = Math.min(varyingKeys.length, 3);
  const pickedKeys = varyingKeys.slice(0, maxKeys);
  return configs.map(c => pickedKeys.map(k => String(c[k])).join(" · "));
}

function normalizeConfigs(configs) {
  const ranges = {};
  METRIC_KEYS.forEach(k => {
    const vals = configs.map(c => c[k]);
    ranges[k] = { min: Math.min(...vals), max: Math.max(...vals) };
  });
  return configs.map(c => {
    const norm = {};
    METRIC_KEYS.forEach(k => {
      const { min, max } = ranges[k];
      const raw = max === min ? 0.5 : (c[k] - min) / (max - min);
      norm[k] = HIGHER_BETTER[k] ? raw : 1 - raw;
    });
    return { ...c, _norm: norm };
  });
}

function fmtMetric(k, v) {
  if (["hit_rate","mrr","ndcg_10","recall_20","faithfulness"].includes(k)) return (v*100).toFixed(1)+"%";
  if (k === "latency_ms") return Math.round(v)+"ms";
  if (k === "cost_per_1k") return "$"+v.toFixed(2);
  return String(v);
}

function Pill({ label, active, onClick, color, small }) {
  return (
    <button onClick={onClick} style={{
      padding: small ? "2px 8px" : "4px 12px", fontSize: small ? 10 : 11, borderRadius: 20,
      border: "1px solid", cursor: "pointer", transition: "all 0.15s", fontWeight: active ? 600 : 400,
      background: active ? (color || "#818cf8") : "transparent",
      color: active ? "#fff" : "#777", borderColor: active ? (color || "#818cf8") : "#333",
      fontFamily: "inherit", whiteSpace: "nowrap",
    }}>{label}</button>
  );
}

function FilterPanel({ data, filters, setFilters }) {
  const [open, setOpen] = useState(false);
  const toggle = (key, val) => {
    const cur = filters[key] || [];
    const next = cur.includes(val) ? cur.filter(v => v !== val) : [...cur, val];
    setFilters({ ...filters, [key]: next.length ? next : undefined });
  };
  const cnt = Object.values(filters).filter(v => v && v.length).length;
  return (
    <div style={{ background: "#0c0c0c", borderRadius: 8, border: "1px solid #1e1e1e", marginBottom: 12 }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", padding: "8px 14px", background: "none", border: "none", color: "#aaa",
        display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", fontSize: 12, fontFamily: "inherit",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span>{open ? "▾" : "▸"}</span> Filters
          {cnt > 0 && <span style={{ background: "#818cf8", color: "#fff", borderRadius: 10, padding: "1px 7px", fontSize: 10 }}>{cnt}</span>}
        </span>
        <span style={{ fontSize: 10, color: "#444" }}>{data.length} runs</span>
      </button>
      {open && (
        <div style={{ padding: "0 14px 12px", display: "flex", flexDirection: "column", gap: 10 }}>
          {PARAM_KEYS.map(key => {
            const vals = getUniqueVals(data, key);
            if (vals.length < 2) return null;
            return (
              <div key={key}>
                <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.08em", color: "#555", marginBottom: 3 }}>{PARAM_LABELS[key]}</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                  {vals.map(v => <Pill key={String(v)} small label={String(v)} active={(filters[key]||[]).includes(v)} onClick={() => toggle(key, v)} />)}
                </div>
              </div>
            );
          })}
          {cnt > 0 && <button onClick={() => setFilters({})} style={{ alignSelf: "flex-start", background: "none", border: "1px solid #333", borderRadius: 6, padding: "3px 10px", color: "#666", cursor: "pointer", fontSize: 10, fontFamily: "inherit" }}>Clear all</button>}
        </div>
      )}
    </div>
  );
}

function ParallelCoords({ configs, labels, highlighted, setHighlighted, topN }) {
  const displayed = topN ? configs.slice(0, topN) : configs;
  const dispLabels = topN ? labels.slice(0, topN) : labels;
  const W = 680, padL = 30, padR = 16, padT = 28, padB = 70;
  const axisGap = (W - padL - padR) / (METRIC_KEYS.length - 1);
  const H = 340;
  const scaleY = (norm) => padT + (1 - norm) * (H - padT - padB);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      {METRIC_KEYS.map((k, i) => {
        const x = padL + i * axisGap;
        return (
          <g key={k}>
            <line x1={x} y1={padT} x2={x} y2={H - padB} stroke="#222" strokeWidth={0.5} />
            <text x={x} y={padT - 8} textAnchor="middle" fill="#888" fontSize={9} fontFamily="'IBM Plex Mono',monospace" fontWeight={500}>{METRIC_LABELS[k]}</text>
            <text x={x} y={padT - 0} textAnchor="middle" fill="#444" fontSize={7} fontFamily="'IBM Plex Mono',monospace">{HIGHER_BETTER[k] ? "▲ better" : "▼ better"}</text>
            <text x={x} y={H - padB + 12} textAnchor="middle" fill="#333" fontSize={7} fontFamily="'IBM Plex Mono',monospace">worst</text>
            <text x={x} y={padT + 8} textAnchor="middle" fill="#333" fontSize={7} fontFamily="'IBM Plex Mono',monospace">best</text>
          </g>
        );
      })}
      {displayed.map((c, ci) => {
        const isHl = highlighted === ci;
        const opacity = highlighted === null ? (displayed.length > 12 ? 0.35 : 0.6) : (isHl ? 1 : 0.08);
        const sw = isHl ? 2.5 : (displayed.length > 12 ? 1 : 1.5);
        const pts = METRIC_KEYS.map((k, i) => {
          const x = padL + i * axisGap;
          const y = scaleY(c._norm[k]);
          return `${x},${y}`;
        }).join(" ");
        return (
          <polyline key={ci} points={pts} fill="none" stroke={COLORS[ci % COLORS.length]}
            strokeWidth={sw} opacity={opacity} strokeLinejoin="round" strokeLinecap="round"
            style={{ cursor: "pointer", transition: "opacity 0.15s" }}
            onMouseEnter={() => setHighlighted(ci)} onMouseLeave={() => setHighlighted(null)} />
        );
      })}
      {highlighted !== null && displayed[highlighted] && METRIC_KEYS.map((k, i) => {
        const c = displayed[highlighted];
        const x = padL + i * axisGap;
        const y = scaleY(c._norm[k]);
        return (
          <g key={k}>
            <circle cx={x} cy={y} r={4} fill={COLORS[highlighted % COLORS.length]} stroke="#000" strokeWidth={1} />
            <text x={x} y={y - 8} textAnchor="middle" fill="#e0e0e0" fontSize={9} fontWeight={600} fontFamily="'IBM Plex Mono',monospace">{fmtMetric(k, c[k])}</text>
          </g>
        );
      })}
      {displayed.length <= 20 && dispLabels.map((lbl, i) => {
        const isHl = highlighted === i;
        return (
          <g key={i} style={{ cursor: "pointer" }} onMouseEnter={() => setHighlighted(i)} onMouseLeave={() => setHighlighted(null)}>
            <rect x={2} y={H - padB + 20 + Math.floor(i / 3) * 14} width={6} height={6} rx={1} fill={COLORS[i % COLORS.length]} opacity={isHl ? 1 : 0.7} />
            <text x={12} y={H - padB + 26 + Math.floor(i / 3) * 14} fill={isHl ? "#fff" : "#666"} fontSize={8} fontFamily="'IBM Plex Mono',monospace" fontWeight={isHl ? 600 : 400}>
              {lbl.length > 28 ? lbl.slice(0, 26) + "…" : lbl}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function RadarOverlay({ configs, labels, highlighted, setHighlighted }) {
  const W = 680, cx = W / 2, cy = 170, R = 130;
  const n = METRIC_KEYS.length;
  const angle = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const ptX = (i, r) => cx + Math.cos(angle(i)) * r;
  const ptY = (i, r) => cy + Math.sin(angle(i)) * r;
  const rings = [0.25, 0.5, 0.75, 1];

  return (
    <svg width="100%" viewBox={`0 0 ${W} 370`} style={{ display: "block" }}>
      {rings.map(r => (
        <polygon key={r} points={METRIC_KEYS.map((_, i) => `${ptX(i, R * r)},${ptY(i, R * r)}`).join(" ")}
          fill="none" stroke="#222" strokeWidth={0.5} />
      ))}
      {METRIC_KEYS.map((k, i) => (
        <g key={k}>
          <line x1={cx} y1={cy} x2={ptX(i, R)} y2={ptY(i, R)} stroke="#1a1a1a" strokeWidth={0.5} />
          <text x={ptX(i, R + 18)} y={ptY(i, R + 18)} textAnchor="middle" dominantBaseline="central"
            fill="#888" fontSize={9} fontWeight={500} fontFamily="'IBM Plex Mono',monospace">{METRIC_LABELS[k]}</text>
        </g>
      ))}
      {configs.map((c, ci) => {
        const isHl = highlighted === ci;
        const opacity = highlighted === null ? 0.25 : (isHl ? 0.4 : 0.04);
        const strokeOp = highlighted === null ? 0.8 : (isHl ? 1 : 0.1);
        const pts = METRIC_KEYS.map((k, i) => `${ptX(i, R * c._norm[k])},${ptY(i, R * c._norm[k])}`).join(" ");
        return (
          <g key={ci} style={{ cursor: "pointer" }} onMouseEnter={() => setHighlighted(ci)} onMouseLeave={() => setHighlighted(null)}>
            <polygon points={pts} fill={COLORS[ci % COLORS.length]} opacity={opacity} stroke={COLORS[ci % COLORS.length]} strokeWidth={isHl ? 2 : 1} strokeOpacity={strokeOp} />
          </g>
        );
      })}
      {highlighted !== null && configs[highlighted] && METRIC_KEYS.map((k, i) => {
        const c = configs[highlighted];
        const px = ptX(i, R * c._norm[k]), py = ptY(i, R * c._norm[k]);
        return <circle key={k} cx={px} cy={py} r={3.5} fill={COLORS[highlighted % COLORS.length]} stroke="#000" strokeWidth={0.8} />;
      })}
      <g transform={`translate(0, 330)`}>
        {labels.map((lbl, i) => (
          <g key={i} style={{ cursor: "pointer" }} onMouseEnter={() => setHighlighted(i)} onMouseLeave={() => setHighlighted(null)}>
            <rect x={20 + (i % 3) * 220} y={Math.floor(i / 3) * 14} width={7} height={7} rx={1.5} fill={COLORS[i % COLORS.length]} opacity={highlighted === i ? 1 : 0.7} />
            <text x={32 + (i % 3) * 220} y={Math.floor(i / 3) * 14 + 6} fill={highlighted === i ? "#fff" : "#666"} fontSize={8} fontFamily="'IBM Plex Mono',monospace" fontWeight={highlighted === i ? 600 : 400}>{lbl.slice(0, 30)}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function HeatmapMatrix({ configs, labels, highlighted, setHighlighted, sortMetric, setSortMetric }) {
  const sorted = [...configs.map((c, i) => ({ ...c, _i: i }))].sort((a, b) => {
    const va = a._norm[sortMetric], vb = b._norm[sortMetric];
    return vb - va;
  });
  const sortedLabels = sorted.map(s => labels[s._i]);
  const sortedOrigIdx = sorted.map(s => s._i);

  const cellColor = (norm) => {
    const r = Math.round(239 * (1 - norm) + 52 * norm);
    const g = Math.round(68 * (1 - norm) + 211 * norm);
    const b = Math.round(68 * (1 - norm) + 153 * norm);
    return `rgb(${r},${g},${b})`;
  };
  const composite = (c) => {
    const vals = METRIC_KEYS.map(k => c._norm[k]);
    return _.mean(vals);
  };

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "flex", gap: 4, marginBottom: 8, alignItems: "center" }}>
        <span style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: "0.06em" }}>Sort by:</span>
        {METRIC_KEYS.map(k => <Pill key={k} small label={METRIC_LABELS[k]} active={sortMetric === k} onClick={() => setSortMetric(k)} />)}
      </div>
      <table style={{ borderCollapse: "separate", borderSpacing: 1, fontSize: 10, fontFamily: "inherit", width: "100%" }}>
        <thead>
          <tr>
            <th style={{ padding: "6px 6px", color: "#555", textAlign: "left", fontSize: 9, minWidth: 80, position: "sticky", left: 0, background: "#0a0a0a", zIndex: 1 }}>#</th>
            <th style={{ padding: "6px 6px", color: "#555", textAlign: "left", fontSize: 9, minWidth: 160, position: "sticky", left: 28, background: "#0a0a0a", zIndex: 1 }}>Config</th>
            <th style={{ padding: "6px 4px", color: "#f59e0b", textAlign: "center", fontSize: 9, fontWeight: 600, minWidth: 48 }}>Score</th>
            {METRIC_KEYS.map(k => (
              <th key={k} style={{ padding: "6px 4px", color: sortMetric === k ? "#818cf8" : "#666", textAlign: "center", fontSize: 9, cursor: "pointer", fontWeight: sortMetric === k ? 700 : 400, minWidth: 52 }}
                onClick={() => setSortMetric(k)}>{METRIC_LABELS[k]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((c, i) => {
            const origI = sortedOrigIdx[i];
            const isHl = highlighted === origI;
            const comp = composite(c);
            return (
              <tr key={i} onMouseEnter={() => setHighlighted(origI)} onMouseLeave={() => setHighlighted(null)}
                style={{ cursor: "pointer", opacity: highlighted === null ? 1 : (isHl ? 1 : 0.4), transition: "opacity 0.1s" }}>
                <td style={{ padding: "5px 6px", color: "#444", position: "sticky", left: 0, background: "#0a0a0a", zIndex: 1 }}>
                  <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: 1.5, background: COLORS[origI % COLORS.length], marginRight: 4, verticalAlign: "middle" }} />
                  {i + 1}
                </td>
                <td style={{ padding: "5px 6px", color: isHl ? "#fff" : "#bbb", fontWeight: isHl ? 600 : 400, position: "sticky", left: 28, background: "#0a0a0a", zIndex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 200 }}>{sortedLabels[i]}</td>
                <td style={{ padding: "5px 4px", textAlign: "center", fontWeight: 700, borderRadius: 3,
                  background: `rgba(250, 204, 21, ${comp * 0.3})`, color: comp > 0.6 ? "#fbbf24" : "#666" }}>
                  {(comp * 100).toFixed(0)}
                </td>
                {METRIC_KEYS.map(k => {
                  const norm = c._norm[k];
                  return (
                    <td key={k} style={{
                      padding: "5px 4px", textAlign: "center", borderRadius: 3, fontWeight: 500,
                      background: cellColor(norm), color: "#fff", fontSize: 10,
                    }}>{fmtMetric(k, c[k])}</td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8, fontSize: 9, color: "#444" }}>
        <span>Worst</span>
        <div style={{ width: 80, height: 8, borderRadius: 4, background: "linear-gradient(to right, rgb(239,68,68), rgb(52,211,153))" }} />
        <span>Best</span>
        <span style={{ marginLeft: 8 }}>All metrics normalized — higher color = better (direction-aware)</span>
      </div>
    </div>
  );
}

function DetailPanel({ config, label, color }) {
  if (!config) return null;
  return (
    <div style={{ background: "#0c0c0c", borderRadius: 8, border: `1px solid ${color}33`, padding: "10px 14px", marginTop: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ width: 10, height: 10, borderRadius: 2, background: color, display: "inline-block" }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: "#e0e0e0" }}>{label}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 6 }}>
        {PARAM_KEYS.map(k => (
          <div key={k} style={{ fontSize: 10 }}>
            <span style={{ color: "#555" }}>{PARAM_LABELS[k]}: </span>
            <span style={{ color: "#ccc" }}>{String(config[k])}</span>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
        {METRIC_KEYS.map(k => {
          const norm = config._norm[k];
          return (
            <div key={k} style={{ background: "#111", borderRadius: 6, padding: "6px 10px", minWidth: 70 }}>
              <div style={{ fontSize: 8, color: "#555", textTransform: "uppercase", marginBottom: 1 }}>{METRIC_LABELS[k]}</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: norm > 0.7 ? "#34d399" : norm > 0.4 ? "#facc15" : "#f87171" }}>{fmtMetric(k, config[k])}</div>
              <div style={{ width: "100%", height: 3, background: "#222", borderRadius: 2, marginTop: 3 }}>
                <div style={{ width: `${norm * 100}%`, height: "100%", background: norm > 0.7 ? "#34d399" : norm > 0.4 ? "#facc15" : "#f87171", borderRadius: 2 }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function RAGSweepAllMetrics() {
  const [filters, setFilters] = useState({});
  const [view, setView] = useState("auto");
  const [highlighted, setHighlighted] = useState(null);
  const [topN, setTopN] = useState(null);
  const [sortMetric, setSortMetric] = useState("hit_rate");

  const filteredData = useMemo(() => {
    return SAMPLE_DATA.filter(d => Object.entries(filters).every(([k, vals]) => !vals || !vals.length || vals.includes(d[k])));
  }, [filters]);

  const configs = useMemo(() => normalizeConfigs(buildConfigs(filteredData)), [filteredData]);
  const labels = useMemo(() => smartLabel(configs), [configs]);
  const count = configs.length;

  const autoView = count <= 5 ? "radar" : count <= 25 ? "parallel" : "heatmap";
  const activeView = view === "auto" ? autoView : view;
  const effectiveTopN = topN || (count > 30 ? 30 : null);

  const views = [
    { id: "auto", label: `Auto (${autoView})` },
    { id: "parallel", label: "Parallel coords" },
    { id: "radar", label: "Radar" },
    { id: "heatmap", label: "Heatmap matrix" },
  ];

  return (
    <div style={{ background: "#050505", color: "#e0e0e0", minHeight: "100vh", fontFamily: "'IBM Plex Mono','SF Mono',monospace", padding: "16px 14px" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: 16, fontWeight: 700, color: "#fff", margin: 0, letterSpacing: "-0.02em" }}>RAG sweep — all metrics view</h1>
          <span style={{ fontSize: 10, color: "#555", borderLeft: "1px solid #333", paddingLeft: 10 }}>
            {count} unique configs · {METRIC_KEYS.length} metrics · normalized (↑ = better)
          </span>
        </div>

        <FilterPanel data={SAMPLE_DATA} filters={filters} setFilters={setFilters} />

        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap", alignItems: "center" }}>
          {views.map(v => <Pill key={v.id} label={v.label} active={view === v.id} onClick={() => setView(v.id)} />)}
          {count > 10 && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 12 }}>
              <span style={{ fontSize: 9, color: "#555", textTransform: "uppercase" }}>Show top</span>
              <input type="range" min={3} max={Math.min(count, 50)} value={effectiveTopN || count}
                onChange={e => setTopN(Number(e.target.value) >= count ? null : Number(e.target.value))}
                style={{ width: 80, accentColor: "#818cf8" }} />
              <span style={{ fontSize: 11, color: "#aaa", minWidth: 24, fontWeight: 600 }}>{effectiveTopN || count}</span>
            </div>
          )}
        </div>

        <div style={{ background: "#0a0a0a", borderRadius: 10, border: "1px solid #1a1a1a", padding: activeView === "heatmap" ? "12px 8px" : 12, overflow: "hidden" }}>
          {activeView === "parallel" && <ParallelCoords configs={configs} labels={labels} highlighted={highlighted} setHighlighted={setHighlighted} topN={effectiveTopN} />}
          {activeView === "radar" && <RadarOverlay configs={configs.slice(0, effectiveTopN || configs.length)} labels={labels.slice(0, effectiveTopN || labels.length)} highlighted={highlighted} setHighlighted={setHighlighted} />}
          {activeView === "heatmap" && <HeatmapMatrix configs={configs} labels={labels} highlighted={highlighted} setHighlighted={setHighlighted} sortMetric={sortMetric} setSortMetric={setSortMetric} />}
        </div>

        <DetailPanel config={highlighted !== null ? configs[highlighted] : null} label={highlighted !== null ? labels[highlighted] : ""} color={highlighted !== null ? COLORS[highlighted % COLORS.length] : "#818cf8"} />

        <div style={{ marginTop: 14, padding: "10px 14px", background: "#0a0a0a", borderRadius: 8, border: "1px solid #1a1a1a", fontSize: 10, color: "#444", lineHeight: 1.7 }}>
          <strong style={{ color: "#666" }}>Reading the chart:</strong> Every metric is normalized so ↑ = better, regardless of whether the raw metric is higher-is-better (hit rate) or lower-is-better (latency, cost).
          Hover any line/polygon/row to isolate a config and see all raw values. The composite score (heatmap) is the mean of all normalized metrics — a quick "overall quality" rank.
          <strong style={{ color: "#666", marginLeft: 8 }}>Smart labels</strong> only show parameters that vary across your filtered set.
          {count > 30 && <span style={{ display: "block", marginTop: 4, color: "#555" }}>Tip: {count} configs is dense — use the "show top N" slider to focus, or add filters to narrow the sweep.</span>}
        </div>
      </div>
    </div>
  );
}
