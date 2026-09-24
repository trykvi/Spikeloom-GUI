$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCommand) {
    $nodeExe = $nodeCommand.Source
} else {
    $nodeExe = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
}

if (-not (Test-Path -LiteralPath $nodeExe)) {
    throw 'Node.js was not found. Install Node.js 20 or newer and run npm install.'
}

$vite = Join-Path $PSScriptRoot '..\node_modules\vite\bin\vite.js'
if (-not (Test-Path -LiteralPath $vite)) {
    throw 'Frontend dependencies are missing. Run npm install (or pnpm install) first.'
}

& $nodeExe $vite --host 127.0.0.1
