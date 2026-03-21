export type BenchPhase = "idle" | "ensuring" | "running" | "complete";
export type SweepPhase = "idle" | "running" | "complete";

export interface ActiveRun {
  status: "running" | "complete" | "error";
  progress: { current: number; total: number };
  error?: string;
}

export interface SweepProgress {
  configIndex: number;
  totalConfigs: number;
  questionIndex: number;
  totalQuestions: number;
}
