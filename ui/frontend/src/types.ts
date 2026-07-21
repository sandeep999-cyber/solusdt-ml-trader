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
