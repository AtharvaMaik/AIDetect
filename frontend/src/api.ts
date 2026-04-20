import type { AnalyzeResponse } from "./types";

const API_URL =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.PROD ? "https://clashking-aidetect.hf.space" : "http://127.0.0.1:7860");

export async function analyzeText(text: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      chunk_tokens: 512,
      overlap_tokens: 64,
      group_size: 3,
    }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Request failed with ${response.status}`);
  }

  return response.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
