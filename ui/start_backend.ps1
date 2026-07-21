# Start the FastAPI replay server.
# Precomputes inference on startup so the first request is instant.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $ProjectRoot

python "$ProjectRoot\run_replay_server.py"
