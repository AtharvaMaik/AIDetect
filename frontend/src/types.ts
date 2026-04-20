export type PredictionLabel = "human" | "ai" | "uncertain";

export type ChunkResult = {
  index: number;
  start_token: number;
  end_token: number;
  token_count: number;
  ai_probability: number;
  human_probability: number;
  prediction: PredictionLabel;
};

export type SuspiciousRegion = {
  start_chunk: number;
  end_chunk: number;
  start_token: number;
  end_token: number;
  ai_probability: number;
  human_probability: number;
  delta_from_document: number;
  reason: string;
};

export type AnalyzeResponse = {
  prediction: PredictionLabel;
  ai_probability: number;
  human_probability: number;
  confidence: number;
  chunks_analyzed: number;
  suspicious_regions: SuspiciousRegion[];
  chunk_results: ChunkResult[];
  model: Record<string, string | number | boolean>;
  notes: string[];
};
