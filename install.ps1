# ds-skiller installer (Windows)
# Renders each skills/ds-*/SKILL.md with this repo's absolute path and writes it into
# every agent's skills folder. Scripts are NOT copied: they always run from this repo,
# so `git pull` updates every agent at once. Re-run after moving the repo.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1              # install
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall   # remove
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Targets "D:\x\skills"
param(
    [switch]$Uninstall,
    [string[]]$Targets
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootFwd = $RepoRoot.Replace([string][char]92, '/')
$SkillsSrc = Join-Path $RepoRoot "skills"
$Utf8 = New-Object System.Text.UTF8Encoding($false)

if (-not $Targets) {
    $Targets = @(Join-Path $env:USERPROFILE ".claude\skills")
    # Optional agents: only if the agent is installed (its home folder exists)
    if (Test-Path (Join-Path $env:USERPROFILE ".codex")) { $Targets += Join-Path $env:USERPROFILE ".codex\skills" }
    if (Test-Path (Join-Path $env:USERPROFILE ".gemini")) { $Targets += Join-Path $env:USERPROFILE ".gemini\antigravity\skills" }
}

$skills = Get-ChildItem -Path $SkillsSrc -Directory -Filter "ds-*"
foreach ($target in $Targets) {
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    foreach ($s in $skills) {
        $dest = Join-Path $target $s.Name
        if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
        if ($Uninstall) { continue }
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        $md = [IO.File]::ReadAllText((Join-Path $s.FullName "SKILL.md"), $Utf8)
        [IO.File]::WriteAllText((Join-Path $dest "SKILL.md"), $md.Replace("{{DS_SKILLER_ROOT}}", $RootFwd), $Utf8)
    }
    $verb = if ($Uninstall) { "Removed" } else { "Installed" }
    Write-Host "$verb $($skills.Count) ds-* skills -> $target" -ForegroundColor Green
}
if (-not $Uninstall) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Write-Warning "uv not found: https://docs.astral.sh/uv/" }
    Write-Host "Repo root: $RootFwd  (run 'uv sync' there once)"
}
