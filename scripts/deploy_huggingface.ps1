param(
    [string]$SpaceId = "clashking/AIDetect"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Checking Hugging Face authentication..."
hf auth whoami

Write-Host "Creating or reusing Hugging Face Docker Space: $SpaceId"
hf repo create $SpaceId --repo-type space --space_sdk docker --exist-ok

if (-not (git remote | Select-String -SimpleMatch "hf-space")) {
    git remote add hf-space "https://huggingface.co/spaces/$SpaceId"
}

Write-Host "Pushing main branch and Git LFS model artifacts to Hugging Face Space..."
git push hf-space main

Write-Host "Space deploy requested: https://huggingface.co/spaces/$SpaceId"
