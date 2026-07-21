import { useState, useEffect, useRef } from 'react'
import type { CheckpointEntry } from '../types'
import { fetchCheckpoints, selectCheckpoint, uploadCheckpoint } from '../api'

interface Props {
  activeRunName: string | null
  onCheckpointChanged: () => void
}

export default function CheckpointSelector({ activeRunName, onCheckpointChanged }: Props) {
  const [checkpoints, setCheckpoints] = useState<CheckpointEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const load = () => {
    setLoading(true)
    setError(null)
    fetchCheckpoints()
      .then(r => setCheckpoints(r.checkpoints.filter(c => c.has_best_pt && !c.is_smoketest)))
      .catch(e => setError(e instanceof Error ? e.message : 'Failed to load'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const handleSelect = (runName: string) => {
    if (runName === activeRunName) return
    setLoading(true)
    selectCheckpoint(runName)
      .then(() => onCheckpointChanged())
      .catch(e => setError(e instanceof Error ? e.message : 'Select failed'))
      .finally(() => setLoading(false))
  }

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    uploadCheckpoint(file)
      .then(() => {
        load()
        onCheckpointChanged()
      })
      .catch(e => setError(e instanceof Error ? e.message : 'Upload failed'))
      .finally(() => setUploading(false))
    // Reset input so same file can be re-selected
    e.target.value = ''
  }

  const fmtMetric = (v: number | undefined | null, decimals = 4) =>
    v !== null && v !== undefined ? v.toFixed(decimals) : '—'

  return (
    <div className="checkpoint-selector">
      <label className="cp-label">Checkpoint</label>

      <select
        className="cp-select"
        value={activeRunName ?? ''}
        disabled={loading || uploading}
        onChange={e => handleSelect(e.target.value)}
      >
        {!activeRunName && <option value="">—</option>}
        {checkpoints.map(c => (
          <option key={c.run_name} value={c.run_name}>
            {c.run_name}
            {c.metrics ? `  (NLL ${fmtMetric(c.metrics.nll)})` : ''}
          </option>
        ))}
      </select>

      <input
        ref={fileRef}
        type="file"
        accept=".pt"
        style={{ display: 'none' }}
        onChange={handleUpload}
      />

      <button
        className="cp-btn"
        disabled={loading || uploading}
        onClick={() => fileRef.current?.click()}
        title="Upload best.pt from local machine"
      >
        {uploading ? '…' : '+'}
      </button>

      <button
        className="cp-btn"
        disabled={loading}
        onClick={load}
        title="Refresh checkpoint list"
      >
        ↻
      </button>

      {error && <span className="cp-error">{error}</span>}
    </div>
  )
}
