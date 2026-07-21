# Start the Vite dev server
$FrontendDir = Join-Path $PSScriptRoot "frontend"
Push-Location $FrontendDir
npm run dev
Pop-Location
