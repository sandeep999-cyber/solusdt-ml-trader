export interface Bar {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  cvd?: number
  vwap_20?: number
  realized_vol?: number
  [key: string]: unknown
}

export interface PredictedState {
  timestamp: string
  price: number
  lower: number
  upper: number
}

export type Decision = 'long' | 'short' | 'flat'

export interface InferenceResult {
  timestamp: string
  window_start: string
  window_end: string
  predicted_future_state: PredictedState[]
  uncertainty: number
  surprise: number
  decision: Decision
}

export interface DecisionEntry {
  timestamp: string
  decision: Decision
}

export interface SeriesResponse {
  symbol: string
  interval: string
  bar_count: number
  bars: Bar[]
}

export interface InferenceRangeResponse {
  symbol: string
  interval: string
  results: InferenceResult[]
  all_decisions: DecisionEntry[]
}

export interface CheckpointEntry {
  run_name: string
  is_smoketest: boolean
  has_best_pt: boolean
  metrics: {
    epoch?: number
    nll?: number
    mse?: number
    baseline_delta?: number
    baseline_loss?: number
    train_loss?: number
  } | null
  config: {
    model_class?: string
    notes?: string
    num_epochs?: number
  }
}

export interface CheckpointListResponse {
  checkpoints: CheckpointEntry[]
}

export interface CheckpointSelectResponse {
  selected: string
  checkpoint: {
    run_name: string
    checkpoint_path: string
    updated: string
  }
}
