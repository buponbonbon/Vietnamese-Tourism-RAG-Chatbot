# Run in PowerShell after extracting this ZIP directly to the repository root.
# Use: .\QUICK_RUN.ps1
# Or copy the individual Python commands from README if local PowerShell policy blocks .ps1.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -u .\ESWA_candidate_set_preflight.py --self-test
if ($LASTEXITCODE -ne 0) { throw 'Self-test failed' }
python -u .\ESWA_candidate_set_preflight.py
if ($LASTEXITCODE -ne 0) { throw 'Corpus mapping failed' }
Write-Host '[DONE] Mapping preflight. Optionally run: python -u ESWA_candidate_set_preflight.py --mine-bm25 --sample 20' 
