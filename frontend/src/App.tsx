import { ChangeEvent, CSSProperties, FormEvent, useEffect, useMemo, useState } from "react";
import { analyzeText, checkHealth } from "./api";
import type { AnalyzeResponse, ChunkResult } from "./types";

const sampleText =
  "The corridor smelled faintly of varnish and old paper, as if time itself had been sealed into the walls. A flickering tube light hummed overhead, casting uneven shadows that stretched and shrank with every pulse of electricity.";

function labelText(label: string) {
  if (label === "ai") return "Likely AI";
  if (label === "human") return "Likely Human";
  return "Uncertain";
}

function scoreTone(aiProbability: number) {
  if (aiProbability >= 70) return "ai";
  if (aiProbability <= 30) return "human";
  return "mixed";
}

function App() {
  const [text, setText] = useState(sampleText);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth().then(setIsHealthy);
  }, []);

  const stats = useMemo(() => {
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    return {
      characters: text.length,
      words,
    };
  }, [text]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    setIsLoading(true);

    try {
      const response = await analyzeText(text);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setIsLoading(false);
    }
  }

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const content = await file.text();
    setText(content);
    setResult(null);
    setError("");
  }

  return (
    <main className="shell">
      <section className="workspace">
        <div className="panel input-panel">
          <div className="brand-row">
            <div>
              <p className="eyebrow">RoBERTa Detector</p>
              <h1>AI Text Analysis</h1>
            </div>
            <div className={`status-dot ${isHealthy ? "ready" : "waiting"}`}>
              <span />
              {isHealthy === null ? "Checking" : isHealthy ? "Ready" : "Offline"}
            </div>
          </div>

          <form onSubmit={onSubmit} className="composer">
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Paste text for analysis"
              spellCheck="true"
            />

            <div className="control-row">
              <label className="file-button">
                <span className="button-icon plus-icon" aria-hidden="true" />
                <input type="file" accept=".txt,text/plain" onChange={onFileChange} />
                TXT
              </label>
              <div className="stat-strip">
                <span>{stats.words.toLocaleString()} words</span>
                <span>{stats.characters.toLocaleString()} chars</span>
              </div>
              <button type="submit" disabled={isLoading || text.trim().length === 0}>
                <span className={`button-icon ${isLoading ? "pulse-icon" : "arrow-icon"}`} aria-hidden="true" />
                {isLoading ? "Analyzing" : "Analyze"}
              </button>
            </div>
          </form>

          {error && <div className="error-line">{error}</div>}
        </div>

        <div className="panel results-panel">
          {!result && !isLoading && (
            <div className="empty-state">
              <div className="scan-visual" aria-hidden="true">
                {Array.from({ length: 18 }).map((_, index) => (
                  <span key={index} />
                ))}
              </div>
              <h2>Awaiting Analysis</h2>
              <p>Results will appear here with document probability, chunk scoring, and high-AI regions.</p>
            </div>
          )}

          {isLoading && (
            <div className="loading-state">
              <div className="loader" />
              <h2>Running RoBERTa</h2>
              <p>Free-tier backends can take a moment after waking.</p>
            </div>
          )}

          {result && <ResultView result={result} />}
        </div>
      </section>
    </main>
  );
}

function ResultView({ result }: { result: AnalyzeResponse }) {
  const tone = scoreTone(result.ai_probability);

  return (
    <div className="result-stack">
      <div className={`verdict ${tone}`}>
        <div>
          <p className="eyebrow">Verdict</p>
          <h2>{labelText(result.prediction)}</h2>
        </div>
        <div className="score-ring" style={{ "--score": `${result.ai_probability}%` } as CSSProperties}>
          <strong>{result.ai_probability.toFixed(1)}%</strong>
          <span>AI</span>
        </div>
      </div>

      <div className="meter-group">
        <Meter label="AI Probability" value={result.ai_probability} tone="ai" />
        <Meter label="Human Probability" value={result.human_probability} tone="human" />
      </div>

      <div className="summary-grid">
        <Metric label="Chunks" value={result.chunks_analyzed.toString()} />
        <Metric label="Confidence" value={`${result.confidence.toFixed(1)}%`} />
        <Metric label="Regions" value={result.suspicious_regions.length.toString()} />
      </div>

      {result.suspicious_regions.length > 0 && (
        <section className="section-block">
          <div className="section-heading">
            <p className="eyebrow">Flagged Regions</p>
            <h3>Unusually High AI Sections</h3>
          </div>
          <div className="region-list">
            {result.suspicious_regions.map((region) => (
              <article className="region-card" key={`${region.start_chunk}-${region.end_chunk}`}>
                <div>
                  <strong>
                    Chunks {region.start_chunk + 1}-{region.end_chunk + 1}
                  </strong>
                  <span>
                    Tokens {region.start_token}-{region.end_token}
                  </span>
                </div>
                <b>{region.ai_probability.toFixed(1)}% AI</b>
              </article>
            ))}
          </div>
        </section>
      )}

      <section className="section-block">
        <div className="section-heading">
          <p className="eyebrow">Chunk Map</p>
          <h3>{result.chunk_results.length} Windows Scanned</h3>
        </div>
        <div className="chunk-map">
          {result.chunk_results.map((chunk) => (
            <ChunkBar chunk={chunk} key={chunk.index} />
          ))}
        </div>
      </section>

      {result.notes.length > 0 && (
        <section className="section-block notes">
          {result.notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </section>
      )}
    </div>
  );
}

function Meter({ label, value, tone }: { label: string; value: number; tone: "ai" | "human" }) {
  return (
    <div className="meter">
      <div className="meter-label">
        <span>{label}</span>
        <strong>{value.toFixed(1)}%</strong>
      </div>
      <div className="track">
        <span className={tone} style={{ width: `${Math.min(value, 100)}%` }} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ChunkBar({ chunk }: { chunk: ChunkResult }) {
  const tone = scoreTone(chunk.ai_probability);
  return (
    <div className={`chunk ${tone}`} title={`Chunk ${chunk.index + 1}: ${chunk.ai_probability}% AI`}>
      <span style={{ height: `${Math.max(8, chunk.ai_probability)}%` }} />
    </div>
  );
}

export default App;
