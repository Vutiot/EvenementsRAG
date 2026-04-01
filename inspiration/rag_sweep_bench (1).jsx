import { useState, useMemo, useCallback } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, ScatterChart, Scatter, ZAxis } from "recharts";

const COLORS = [
  "#6366f1","#06b6d4","#f59e0b","#ef4444","#10b981","#ec4899",
  "#8b5cf6","#14b8a6","#f97316","#64748b","#a855f7","#0ea5e9"
];

const SAMPLE_DATA = [
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"OpenAI-3-small",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:5,hit_rate:0.72,mrr:0.65,ndcg_10:0.58,recall_20:0.78,faithfulness:0.71,latency_ms:210,cost_per_1k:0.12},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"OpenAI-3-small",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:5,hit_rate:0.81,mrr:0.74,ndcg_10:0.67,recall_20:0.85,faithfulness:0.78,latency_ms:380,cost_per_1k:0.18},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:5,hit_rate:0.78,mrr:0.71,ndcg_10:0.64,recall_20:0.83,faithfulness:0.75,latency_ms:290,cost_per_1k:0.15},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:5,hit_rate:0.87,mrr:0.81,ndcg_10:0.74,recall_20:0.91,faithfulness:0.83,latency_ms:460,cost_per_1k:0.22},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:5,hit_rate:0.80,mrr:0.73,ndcg_10:0.66,recall_20:0.85,faithfulness:0.76,latency_ms:195,cost_per_1k:0.14},
  {chunk_size:256,overlap:0,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"VoyageRerank-2.5",hybrid_method:"none",top_k:5,hit_rate:0.89,mrr:0.83,ndcg_10:0.76,recall_20:0.93,faithfulness:0.85,latency_ms:390,cost_per_1k:0.21},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-small",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.76,mrr:0.68,ndcg_10:0.62,recall_20:0.82,faithfulness:0.74,latency_ms:230,cost_per_1k:0.13},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-small",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.85,mrr:0.78,ndcg_10:0.71,recall_20:0.89,faithfulness:0.81,latency_ms:410,cost_per_1k:0.19},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.82,mrr:0.75,ndcg_10:0.68,recall_20:0.87,faithfulness:0.79,latency_ms:310,cost_per_1k:0.16},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.90,mrr:0.84,ndcg_10:0.77,recall_20:0.94,faithfulness:0.86,latency_ms:490,cost_per_1k:0.23},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"bge-reranker-large",hybrid_method:"none",top_k:10,hit_rate:0.88,mrr:0.82,ndcg_10:0.75,recall_20:0.92,faithfulness:0.84,latency_ms:350,cost_per_1k:0.17},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.84,mrr:0.77,ndcg_10:0.70,recall_20:0.88,faithfulness:0.80,latency_ms:215,cost_per_1k:0.15},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"VoyageRerank-2.5",hybrid_method:"none",top_k:10,hit_rate:0.92,mrr:0.86,ndcg_10:0.79,recall_20:0.95,faithfulness:0.88,latency_ms:410,cost_per_1k:0.22},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"JinaAI-v2",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.83,mrr:0.76,ndcg_10:0.69,recall_20:0.87,faithfulness:0.78,latency_ms:180,cost_per_1k:0.11},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"JinaAI-v2",context_embedding:"none",retrieval:"dense",reranker:"bge-reranker-large",hybrid_method:"none",top_k:10,hit_rate:0.93,mrr:0.87,ndcg_10:0.81,recall_20:0.96,faithfulness:0.89,latency_ms:340,cost_per_1k:0.16},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"contextual",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.88,mrr:0.82,ndcg_10:0.75,recall_20:0.92,faithfulness:0.84,latency_ms:320,cost_per_1k:0.28},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"contextual",retrieval:"hybrid",reranker:"none",hybrid_method:"RRF-k60",top_k:10,hit_rate:0.91,mrr:0.85,ndcg_10:0.78,recall_20:0.95,faithfulness:0.87,latency_ms:380,cost_per_1k:0.30},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"contextual",retrieval:"hybrid",reranker:"CohereRerank",hybrid_method:"RRF-k60",top_k:10,hit_rate:0.95,mrr:0.91,ndcg_10:0.85,recall_20:0.98,faithfulness:0.92,latency_ms:520,cost_per_1k:0.36},
  {chunk_size:512,overlap:10,chunk_method:"semantic",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.84,mrr:0.77,ndcg_10:0.70,recall_20:0.89,faithfulness:0.81,latency_ms:340,cost_per_1k:0.18},
  {chunk_size:512,overlap:10,chunk_method:"semantic",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.91,mrr:0.85,ndcg_10:0.78,recall_20:0.94,faithfulness:0.87,latency_ms:510,cost_per_1k:0.25},
  {chunk_size:512,overlap:10,chunk_method:"late_chunking",embedding:"JinaAI-v2",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.89,mrr:0.83,ndcg_10:0.77,recall_20:0.93,faithfulness:0.85,latency_ms:200,cost_per_1k:0.12},
  {chunk_size:512,overlap:10,chunk_method:"late_chunking",embedding:"JinaAI-v2",context_embedding:"none",retrieval:"dense",reranker:"bge-reranker-large",hybrid_method:"none",top_k:10,hit_rate:0.94,mrr:0.89,ndcg_10:0.83,recall_20:0.97,faithfulness:0.91,latency_ms:360,cost_per_1k:0.17},
  {chunk_size:1024,overlap:20,chunk_method:"recursive",embedding:"OpenAI-3-small",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.74,mrr:0.66,ndcg_10:0.59,recall_20:0.80,faithfulness:0.73,latency_ms:250,cost_per_1k:0.14},
  {chunk_size:1024,overlap:20,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.80,mrr:0.73,ndcg_10:0.66,recall_20:0.85,faithfulness:0.78,latency_ms:330,cost_per_1k:0.17},
  {chunk_size:1024,overlap:20,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.88,mrr:0.82,ndcg_10:0.75,recall_20:0.92,faithfulness:0.85,latency_ms:510,cost_per_1k:0.24},
  {chunk_size:1024,overlap:20,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"none",retrieval:"dense",reranker:"VoyageRerank-2.5",hybrid_method:"none",top_k:10,hit_rate:0.90,mrr:0.84,ndcg_10:0.77,recall_20:0.94,faithfulness:0.87,latency_ms:430,cost_per_1k:0.23},
  {chunk_size:1024,overlap:20,chunk_method:"page_level",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.79,mrr:0.72,ndcg_10:0.65,recall_20:0.84,faithfulness:0.77,latency_ms:300,cost_per_1k:0.16},
  {chunk_size:1024,overlap:20,chunk_method:"page_level",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.87,mrr:0.81,ndcg_10:0.74,recall_20:0.91,faithfulness:0.84,latency_ms:480,cost_per_1k:0.23},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"BGE-M3",context_embedding:"none",retrieval:"hybrid",reranker:"none",hybrid_method:"RRF-k60",top_k:10,hit_rate:0.87,mrr:0.81,ndcg_10:0.74,recall_20:0.91,faithfulness:0.83,latency_ms:260,cost_per_1k:0.10},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"BGE-M3",context_embedding:"none",retrieval:"hybrid",reranker:"bge-reranker-large",hybrid_method:"RRF-k60",top_k:10,hit_rate:0.92,mrr:0.86,ndcg_10:0.80,recall_20:0.96,faithfulness:0.89,latency_ms:400,cost_per_1k:0.15},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"BGE-M3",context_embedding:"none",retrieval:"hybrid",reranker:"none",hybrid_method:"SPLADE+dense",top_k:10,hit_rate:0.89,mrr:0.83,ndcg_10:0.76,recall_20:0.93,faithfulness:0.85,latency_ms:280,cost_per_1k:0.11},
  {chunk_size:512,overlap:20,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"none",hybrid_method:"none",top_k:10,hit_rate:0.83,mrr:0.76,ndcg_10:0.69,recall_20:0.88,faithfulness:0.80,latency_ms:315,cost_per_1k:0.16},
  {chunk_size:512,overlap:20,chunk_method:"recursive",embedding:"OpenAI-3-large",context_embedding:"none",retrieval:"dense",reranker:"CohereRerank",hybrid_method:"none",top_k:10,hit_rate:0.91,mrr:0.85,ndcg_10:0.78,recall_20:0.94,faithfulness:0.87,latency_ms:495,cost_per_1k:0.23},
  {chunk_size:512,overlap:10,chunk_method:"recursive",embedding:"Voyage-2",context_embedding:"contextual",retrieval:"hybrid",reranker:"VoyageRerank-2.5",hybrid_method:"RRF-k60",top_k:20,hit_rate:0.96,mrr:0.92,ndcg_10:0.87,recall_20:0.98,faithfulness:0.93,latency_ms:580,cost_per_1k:0.42},
];

const PARAM_KEYS = ["chunk_size","overlap","chunk_method","embedding","context_embedding","retrieval","reranker","hybrid_method","top_k"];
const METRIC_KEYS = ["hit_rate","mrr","ndcg_10","recall_20","faithfulness","latency_ms","cost_per_1k"];

const PARAM_LABELS = {
  chunk_size:"Chunk size",overlap:"Overlap %",chunk_method:"Chunk method",
  embedding:"Embedding model",context_embedding:"Context embedding",
  retrieval:"Retrieval type",reranker:"Reranker",hybrid_method:"Hybrid method",top_k:"Top-K"
};
const METRIC_LABELS = {
  hit_rate:"Hit rate",mrr:"MRR",ndcg_10:"NDCG@10",recall_20:"Recall@20",
  faithfulness:"Faithfulness",latency_ms:"Latency (ms)",cost_per_1k:"Cost/1k queries ($)"
};
const HIGHER_BETTER = {hit_rate:true,mrr:true,ndcg_10:true,recall_20:true,faithfulness:true,latency_ms:false,cost_per_1k:false};

function getUniqueValues(data, key) {
  return [...new Set(data.map(d => d[key]))].sort((a,b) => typeof a === "number" ? a - b : String(a).localeCompare(String(b)));
}

function Pill({ label, active, onClick, color }) {
  return (
    <button onClick={onClick} style={{
      padding:"4px 12px",fontSize:12,borderRadius:20,border:"1px solid",cursor:"pointer",
      fontFamily:"'IBM Plex Mono',monospace",transition:"all 0.15s",fontWeight:active?600:400,
      background:active?(color||"#6366f1"):("transparent"),
      color:active?"#fff":"#888",
      borderColor:active?(color||"#6366f1"):"#333",
    }}>{label}</button>
  );
}

function Select({ label, value, onChange, options, style }) {
  return (
    <div style={{display:"flex",flexDirection:"column",gap:4,...style}}>
      <label style={{fontSize:10,textTransform:"uppercase",letterSpacing:"0.08em",color:"#666",fontFamily:"'IBM Plex Mono',monospace"}}>{label}</label>
      <select value={value} onChange={e=>onChange(e.target.value)} style={{
        background:"#111",color:"#e0e0e0",border:"1px solid #333",borderRadius:6,padding:"6px 10px",
        fontSize:13,fontFamily:"'IBM Plex Mono',monospace",cursor:"pointer",
      }}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function FilterPanel({ data, filters, setFilters }) {
  const [expanded, setExpanded] = useState(false);
  const toggleFilter = (key, val) => {
    const cur = filters[key] || [];
    const next = cur.includes(val) ? cur.filter(v=>v!==val) : [...cur, val];
    setFilters({...filters, [key]: next.length ? next : undefined});
  };
  const activeCount = Object.values(filters).filter(v=>v&&v.length).length;
  return (
    <div style={{background:"#0a0a0a",borderRadius:8,border:"1px solid #222",overflow:"hidden"}}>
      <button onClick={()=>setExpanded(!expanded)} style={{
        width:"100%",padding:"10px 16px",background:"none",border:"none",color:"#ccc",
        display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",fontSize:13,
        fontFamily:"'IBM Plex Mono',monospace",
      }}>
        <span style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:14}}>&#x25BC;</span> Filters
          {activeCount > 0 && <span style={{background:"#6366f1",color:"#fff",borderRadius:10,padding:"1px 8px",fontSize:11}}>{activeCount}</span>}
        </span>
        <span style={{fontSize:11,color:"#555"}}>{expanded?"collapse":"expand"}</span>
      </button>
      {expanded && (
        <div style={{padding:"0 16px 14px",display:"flex",flexDirection:"column",gap:12}}>
          {PARAM_KEYS.map(key => {
            const vals = getUniqueValues(data, key);
            if (vals.length < 2) return null;
            const active = filters[key] || [];
            return (
              <div key={key}>
                <div style={{fontSize:10,textTransform:"uppercase",letterSpacing:"0.08em",color:"#555",marginBottom:4,fontFamily:"'IBM Plex Mono',monospace"}}>{PARAM_LABELS[key]}</div>
                <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
                  {vals.map(v => <Pill key={String(v)} label={String(v)} active={active.includes(v)} onClick={()=>toggleFilter(key,v)} />)}
                </div>
              </div>
            );
          })}
          {activeCount > 0 && <button onClick={()=>setFilters({})} style={{alignSelf:"flex-start",background:"none",border:"1px solid #333",borderRadius:6,padding:"4px 12px",color:"#888",cursor:"pointer",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}}>Clear all filters</button>}
        </div>
      )}
    </div>
  );
}

function DataTable({ data, metric, xAxis, groupBy }) {
  const sorted = [...data].sort((a,b) => HIGHER_BETTER[metric] ? b[metric]-a[metric] : a[metric]-b[metric]);
  const best = sorted[0]?.[metric];
  const worst = sorted[sorted.length-1]?.[metric];
  return (
    <div style={{overflowX:"auto",borderRadius:8,border:"1px solid #222"}}>
      <table style={{width:"100%",borderCollapse:"collapse",fontSize:12,fontFamily:"'IBM Plex Mono',monospace"}}>
        <thead>
          <tr style={{background:"#111",borderBottom:"1px solid #333"}}>
            <th style={{padding:"10px 12px",textAlign:"left",color:"#888",fontWeight:500,fontSize:10,textTransform:"uppercase",letterSpacing:"0.05em"}}>#</th>
            {PARAM_KEYS.map(k => <th key={k} style={{padding:"10px 8px",textAlign:"left",color:k===xAxis||k===groupBy?"#6366f1":"#888",fontWeight:500,fontSize:10,textTransform:"uppercase",letterSpacing:"0.05em",whiteSpace:"nowrap"}}>{PARAM_LABELS[k]}</th>)}
            {METRIC_KEYS.map(k => <th key={k} style={{padding:"10px 8px",textAlign:"right",color:k===metric?"#6366f1":"#888",fontWeight:k===metric?700:500,fontSize:10,textTransform:"uppercase",letterSpacing:"0.05em",whiteSpace:"nowrap"}}>{METRIC_LABELS[k]}</th>)}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr key={i} style={{borderBottom:"1px solid #1a1a1a",background:i%2===0?"transparent":"#0a0a0a"}}>
              <td style={{padding:"8px 12px",color:"#444"}}>{i+1}</td>
              {PARAM_KEYS.map(k => <td key={k} style={{padding:"8px",color:"#ccc",whiteSpace:"nowrap"}}>{String(row[k])}</td>)}
              {METRIC_KEYS.map(k => {
                const val = row[k];
                const isBest = k===metric && val===best;
                const isWorst = k===metric && val===worst;
                return <td key={k} style={{
                  padding:"8px",textAlign:"right",whiteSpace:"nowrap",fontWeight:k===metric?600:400,
                  color:isBest?(HIGHER_BETTER[k]?"#10b981":"#ef4444"):isWorst?(HIGHER_BETTER[k]?"#ef4444":"#10b981"):k===metric?"#e0e0e0":"#888",
                }}>{k.includes("rate")||k==="mrr"||k.includes("ndcg")||k.includes("recall")||k==="faithfulness"?(val*100).toFixed(1)+"%":k==="latency_ms"?val+"ms":k==="cost_per_1k"?"$"+val.toFixed(2):val}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BarView({ data, metric, xAxis, groupBy }) {
  const groups = groupBy && groupBy !== xAxis ? getUniqueValues(data, groupBy) : [null];
  const xVals = getUniqueValues(data, xAxis);
  const chartData = xVals.map(xv => {
    const entry = { name: String(xv) };
    groups.forEach((gv, gi) => {
      const subset = data.filter(d => String(d[xAxis]) === String(xv) && (gv === null || String(d[groupBy]) === String(gv)));
      const avg = subset.length ? subset.reduce((s, d) => s + d[metric], 0) / subset.length : 0;
      entry[gv === null ? metric : String(gv)] = Math.round(avg * 1000) / 1000;
    });
    return entry;
  });
  const barKeys = groups.map(g => g === null ? metric : String(g));
  const fmt = (v) => {
    if (metric.includes("rate")||metric==="mrr"||metric.includes("ndcg")||metric.includes("recall")||metric==="faithfulness") return (v*100).toFixed(1)+"%";
    if (metric==="latency_ms") return v+"ms";
    if (metric==="cost_per_1k") return "$"+v.toFixed(2);
    return v;
  };
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={chartData} margin={{top:10,right:10,left:10,bottom:40}}>
        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
        <XAxis dataKey="name" tick={{fill:"#888",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} angle={-35} textAnchor="end" interval={0} />
        <YAxis tick={{fill:"#888",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} tickFormatter={fmt} />
        <Tooltip contentStyle={{background:"#111",border:"1px solid #333",borderRadius:8,fontSize:12,fontFamily:"'IBM Plex Mono',monospace"}} labelStyle={{color:"#888"}} formatter={(v)=>fmt(v)} />
        {barKeys.length > 1 && <Legend wrapperStyle={{fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} />}
        {barKeys.map((k, i) => <Bar key={k} dataKey={k} fill={COLORS[i % COLORS.length]} radius={[3,3,0,0]} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

function LineView({ data, metric, xAxis, groupBy }) {
  const groups = groupBy && groupBy !== xAxis ? getUniqueValues(data, groupBy) : [null];
  const xVals = getUniqueValues(data, xAxis);
  const chartData = xVals.map(xv => {
    const entry = { name: typeof xv === "number" ? xv : String(xv) };
    groups.forEach(gv => {
      const subset = data.filter(d => String(d[xAxis]) === String(xv) && (gv === null || String(d[groupBy]) === String(gv)));
      const avg = subset.length ? subset.reduce((s, d) => s + d[metric], 0) / subset.length : null;
      entry[gv === null ? metric : String(gv)] = avg !== null ? Math.round(avg * 1000) / 1000 : null;
    });
    return entry;
  });
  const lineKeys = groups.map(g => g === null ? metric : String(g));
  const fmt = (v) => {
    if (v === null) return "—";
    if (metric.includes("rate")||metric==="mrr"||metric.includes("ndcg")||metric.includes("recall")||metric==="faithfulness") return (v*100).toFixed(1)+"%";
    if (metric==="latency_ms") return v+"ms";
    if (metric==="cost_per_1k") return "$"+v.toFixed(2);
    return v;
  };
  return (
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={chartData} margin={{top:10,right:10,left:10,bottom:40}}>
        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
        <XAxis dataKey="name" tick={{fill:"#888",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} angle={-35} textAnchor="end" interval={0} />
        <YAxis tick={{fill:"#888",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} tickFormatter={fmt} />
        <Tooltip contentStyle={{background:"#111",border:"1px solid #333",borderRadius:8,fontSize:12,fontFamily:"'IBM Plex Mono',monospace"}} formatter={(v)=>fmt(v)} />
        {lineKeys.length > 1 && <Legend wrapperStyle={{fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} />}
        {lineKeys.map((k, i) => <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{r:4,fill:COLORS[i%COLORS.length]}} connectNulls />)}
      </LineChart>
    </ResponsiveContainer>
  );
}

function HeatmapView({ data, metric, xAxis, groupBy }) {
  if (!groupBy || groupBy === xAxis) return <div style={{padding:40,textAlign:"center",color:"#555",fontFamily:"'IBM Plex Mono',monospace",fontSize:13}}>Select a different group-by parameter for the heatmap Y-axis</div>;
  const xVals = getUniqueValues(data, xAxis);
  const yVals = getUniqueValues(data, groupBy);
  const vals = [];
  const grid = yVals.map(yv => {
    return xVals.map(xv => {
      const subset = data.filter(d => String(d[xAxis]) === String(xv) && String(d[groupBy]) === String(yv));
      const avg = subset.length ? subset.reduce((s, d) => s + d[metric], 0) / subset.length : null;
      if (avg !== null) vals.push(avg);
      return avg;
    });
  });
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const norm = (v) => max === min ? 0.5 : (v - min) / (max - min);
  const hb = HIGHER_BETTER[metric];
  const cellColor = (v) => {
    if (v === null) return "#111";
    const t = norm(v);
    const intensity = hb ? t : 1 - t;
    const r = Math.round(239 * (1 - intensity) + 16 * intensity);
    const g = Math.round(68 * (1 - intensity) + 185 * intensity);
    const b = Math.round(68 * (1 - intensity) + 129 * intensity);
    return `rgb(${r},${g},${b})`;
  };
  const fmt = (v) => {
    if (v === null) return "—";
    if (metric.includes("rate")||metric==="mrr"||metric.includes("ndcg")||metric.includes("recall")||metric==="faithfulness") return (v*100).toFixed(1);
    if (metric==="latency_ms") return Math.round(v);
    if (metric==="cost_per_1k") return v.toFixed(2);
    return Math.round(v*100)/100;
  };
  const cellW = Math.max(56, Math.min(90, (640 - 140) / xVals.length));
  return (
    <div style={{overflowX:"auto"}}>
      <table style={{borderCollapse:"separate",borderSpacing:2,fontFamily:"'IBM Plex Mono',monospace",fontSize:11}}>
        <thead>
          <tr>
            <th style={{padding:"6px 8px",color:"#555",textAlign:"left",fontSize:10,minWidth:130}}>{PARAM_LABELS[groupBy]} \ {PARAM_LABELS[xAxis]}</th>
            {xVals.map(xv => <th key={String(xv)} style={{padding:"6px 4px",color:"#888",textAlign:"center",fontSize:10,minWidth:cellW,maxWidth:cellW,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{String(xv)}</th>)}
          </tr>
        </thead>
        <tbody>
          {yVals.map((yv, yi) => (
            <tr key={String(yv)}>
              <td style={{padding:"6px 8px",color:"#aaa",whiteSpace:"nowrap",fontSize:11}}>{String(yv)}</td>
              {xVals.map((xv, xi) => {
                const v = grid[yi][xi];
                return (
                  <td key={String(xv)} style={{
                    padding:"8px 4px",textAlign:"center",borderRadius:4,fontWeight:600,fontSize:12,
                    background:cellColor(v),color:v!==null?"#fff":"#333",minWidth:cellW,
                  }}>{fmt(v)}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{display:"flex",alignItems:"center",gap:8,marginTop:10,fontSize:10,color:"#555"}}>
        <span>{hb?"worst":"best"}</span>
        <div style={{width:120,height:10,borderRadius:5,background:"linear-gradient(to right, rgb(239,68,68), rgb(16,185,129)"}} />
        <span>{hb?"best":"worst"}</span>
        <span style={{marginLeft:8}}>{METRIC_LABELS[metric]}</span>
      </div>
    </div>
  );
}

function ScatterView({ data, metric, xAxis, groupBy }) {
  const secondMetric = METRIC_KEYS.find(k => k !== metric) || metric;
  const [yMetric, setYMetric] = useState(metric === "latency_ms" ? "hit_rate" : "latency_ms");
  const groups = groupBy ? getUniqueValues(data, groupBy) : [null];
  const fmt = (v, k) => {
    if (k.includes("rate")||k==="mrr"||k.includes("ndcg")||k.includes("recall")||k==="faithfulness") return (v*100).toFixed(1)+"%";
    if (k==="latency_ms") return v+"ms";
    if (k==="cost_per_1k") return "$"+v.toFixed(2);
    return v;
  };
  return (
    <div>
      <div style={{display:"flex",gap:8,marginBottom:10,alignItems:"center"}}>
        <span style={{fontSize:10,color:"#555",fontFamily:"'IBM Plex Mono',monospace",textTransform:"uppercase"}}>Y-axis:</span>
        {METRIC_KEYS.filter(k=>k!==metric).map(k => (
          <Pill key={k} label={METRIC_LABELS[k]} active={yMetric===k} onClick={()=>setYMetric(k)} color="#06b6d4" />
        ))}
      </div>
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{top:10,right:10,left:10,bottom:20}}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis dataKey="x" name={METRIC_LABELS[metric]} tick={{fill:"#888",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} tickFormatter={v=>fmt(v,metric)} label={{value:METRIC_LABELS[metric],fill:"#666",fontSize:11,dy:14}} />
          <YAxis dataKey="y" name={METRIC_LABELS[yMetric]} tick={{fill:"#888",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} tickFormatter={v=>fmt(v,yMetric)} label={{value:METRIC_LABELS[yMetric],fill:"#666",fontSize:11,angle:-90,dx:-16}} />
          <ZAxis range={[60,60]} />
          <Tooltip cursor={{strokeDasharray:'3 3'}} contentStyle={{background:"#111",border:"1px solid #333",borderRadius:8,fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}}
            formatter={(v,name)=> name===METRIC_LABELS[metric]?fmt(v,metric):fmt(v,yMetric)} />
          {groups.length > 1 && <Legend wrapperStyle={{fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}} />}
          {groups.map((gv, gi) => {
            const subset = gv === null ? data : data.filter(d=>String(d[groupBy])===String(gv));
            const pts = subset.map(d=>({x:d[metric],y:d[yMetric],name:d.embedding+" | "+d.reranker}));
            return <Scatter key={String(gv)||"all"} name={gv===null?"all":String(gv)} data={pts} fill={COLORS[gi%COLORS.length]} />;
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function Stats({ data, metric }) {
  if (!data.length) return null;
  const vals = data.map(d => d[metric]);
  const mean = vals.reduce((s,v) => s+v, 0) / vals.length;
  const best = HIGHER_BETTER[metric] ? Math.max(...vals) : Math.min(...vals);
  const worst = HIGHER_BETTER[metric] ? Math.min(...vals) : Math.max(...vals);
  const fmt = (v) => {
    if (metric.includes("rate")||metric==="mrr"||metric.includes("ndcg")||metric.includes("recall")||metric==="faithfulness") return (v*100).toFixed(1)+"%";
    if (metric==="latency_ms") return Math.round(v)+"ms";
    if (metric==="cost_per_1k") return "$"+v.toFixed(2);
    return v;
  };
  const cards = [
    {label:"Runs",value:data.length,color:"#6366f1"},
    {label:"Best "+METRIC_LABELS[metric],value:fmt(best),color:"#10b981"},
    {label:"Mean",value:fmt(mean),color:"#f59e0b"},
    {label:"Worst",value:fmt(worst),color:"#ef4444"},
  ];
  return (
    <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8,marginBottom:16}}>
      {cards.map(c => (
        <div key={c.label} style={{background:"#0a0a0a",borderRadius:8,padding:"10px 14px",borderLeft:`3px solid ${c.color}`}}>
          <div style={{fontSize:10,color:"#555",textTransform:"uppercase",letterSpacing:"0.06em",fontFamily:"'IBM Plex Mono',monospace",marginBottom:2}}>{c.label}</div>
          <div style={{fontSize:18,fontWeight:600,color:"#e0e0e0",fontFamily:"'IBM Plex Mono',monospace"}}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}

export default function RAGSweepBench() {
  const [data] = useState(SAMPLE_DATA);
  const [view, setView] = useState("bar");
  const [metric, setMetric] = useState("hit_rate");
  const [xAxis, setXAxis] = useState("embedding");
  const [groupBy, setGroupBy] = useState("reranker");
  const [filters, setFilters] = useState({});

  const filteredData = useMemo(() => {
    return data.filter(d => {
      return Object.entries(filters).every(([key, vals]) => {
        if (!vals || !vals.length) return true;
        return vals.includes(d[key]);
      });
    });
  }, [data, filters]);

  const views = [
    {id:"bar",label:"Bar chart"},
    {id:"line",label:"Line chart"},
    {id:"heatmap",label:"Heatmap"},
    {id:"scatter",label:"Scatter"},
    {id:"table",label:"Table"},
  ];

  return (
    <div style={{
      background:"#050505",color:"#e0e0e0",minHeight:"100vh",
      fontFamily:"'IBM Plex Mono','SF Mono',monospace",padding:"20px 16px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

      <div style={{maxWidth:960,margin:"0 auto"}}>
        <div style={{marginBottom:20,display:"flex",alignItems:"baseline",gap:12,flexWrap:"wrap"}}>
          <h1 style={{fontSize:18,fontWeight:700,color:"#fff",margin:0,letterSpacing:"-0.02em"}}>RAG sweep bench</h1>
          <span style={{fontSize:11,color:"#555",borderLeft:"1px solid #333",paddingLeft:12}}>
            {filteredData.length} runs · {PARAM_KEYS.length} parameters · {METRIC_KEYS.length} metrics
          </span>
        </div>

        <FilterPanel data={data} filters={filters} setFilters={setFilters} />

        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,margin:"14px 0"}}>
          <Select label="X-axis / primary parameter" value={xAxis}
            onChange={setXAxis}
            options={PARAM_KEYS.map(k=>({value:k,label:PARAM_LABELS[k]}))} />
          <Select label="Metric" value={metric}
            onChange={setMetric}
            options={METRIC_KEYS.map(k=>({value:k,label:METRIC_LABELS[k]+(HIGHER_BETTER[k]?" ↑":" ↓")}))} />
          <Select label="Group by / color" value={groupBy}
            onChange={setGroupBy}
            options={[{value:"",label:"None"},...PARAM_KEYS.filter(k=>k!==xAxis).map(k=>({value:k,label:PARAM_LABELS[k]}))]} />
        </div>

        <Stats data={filteredData} metric={metric} />

        <div style={{display:"flex",gap:4,marginBottom:14}}>
          {views.map(v => <Pill key={v.id} label={v.label} active={view===v.id} onClick={()=>setView(v.id)} />)}
        </div>

        <div style={{background:"#0a0a0a",borderRadius:10,border:"1px solid #1a1a1a",padding:view==="table"?0:16,overflow:"hidden"}}>
          {view === "bar" && <BarView data={filteredData} metric={metric} xAxis={xAxis} groupBy={groupBy} />}
          {view === "line" && <LineView data={filteredData} metric={metric} xAxis={xAxis} groupBy={groupBy} />}
          {view === "heatmap" && <HeatmapView data={filteredData} metric={metric} xAxis={xAxis} groupBy={groupBy} />}
          {view === "scatter" && <ScatterView data={filteredData} metric={metric} xAxis={xAxis} groupBy={groupBy} />}
          {view === "table" && <DataTable data={filteredData} metric={metric} xAxis={xAxis} groupBy={groupBy} />}
        </div>

        <div style={{marginTop:16,padding:"12px 16px",background:"#0a0a0a",borderRadius:8,border:"1px solid #1a1a1a",fontSize:11,color:"#555",lineHeight:1.6}}>
          <strong style={{color:"#888"}}>How to use:</strong> Select a primary parameter (X-axis), a metric to measure, and optionally a group-by dimension to compare slices.
          The heatmap view requires a group-by selection to create the Y-axis. Scatter view lets you pick a second metric for the Y-axis to explore tradeoffs (e.g. hit_rate vs latency).
          Use filters to narrow the sweep to specific configurations. The table view sorts by your selected metric, highlighting best/worst values.
          <span style={{display:"block",marginTop:6,color:"#444"}}>Sample data: 34 runs from synthesized production RAG benchmarks (Anthropic, LlamaIndex, Agentset, Milvus, NVIDIA sources). Replace with your own sweep JSON.</span>
        </div>
      </div>
    </div>
  );
}
