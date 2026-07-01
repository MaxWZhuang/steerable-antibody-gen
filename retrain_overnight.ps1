# retrain_overnight.ps1
# Re-preprocess ASD, retrain the antigen stages on the corrected data, and
# generate candidates. Reuses the existing OAS/paired foundation, since OAS
# data is unchanged by the fixes (the antigen configs already init from it).
#
# Run from the repo root:  .\retrain_overnight.ps1
# Stops at the first failing stage so a bad step never cascades into training.
# Each stage logs to logs\<timestamp>_<stage>.log.

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path logs, outputs | Out-Null

$steps = @(
    @{ Name = "preprocess_asd"; Args = @(
        "scripts/prepare_antibody_antigen.py"
        # add --input <dir> if your raw ASD parquet is not under data/raw/asd-antibody-antigen
    )},
    @{ Name = "antigen_reallabel"; Args = @(
        "scripts/mlm_train.py", "--config", "configs/refine_antigen_real_label.yaml",
        "--no-resume-from-last", "--warmup-steps", "300", "--no-progress"
    )},
    @{ Name = "antigen_infill"; Args = @(
        "scripts/mlm_train.py", "--config", "configs/refine_antigen_hcdr3_infill.yaml",
        "--no-resume-from-last", "--warmup-steps", "200", "--no-progress"
    )},
    @{ Name = "generate"; Args = @(
        "scripts/hcdr3_infill.py",
        "--checkpoint", "checkpoints/mlm_antigen_hcdr3_infill_refine/best.pt",
        "--data-path", "data/processed/antibody_antigen/antibody_antigen.jsonl.gz",
        "--split", "val", "--num-records", "20", "--num-samples", "16",
        "--length-mode", "empirical",
        "--score-checkpoint", "checkpoints/mlm_antigen_real_label_refine/best.pt",
        "--device", "cuda",
        "--output-path", "outputs/hcdr3_candidates.jsonl"
    )}
)

foreach ($s in $steps) {
    $log = "logs/${ts}_$($s.Name).log"
    Write-Host "=== [$($s.Name)] starting -> $log ==="
    & python @($s.Args) *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! [$($s.Name)] FAILED (exit $LASTEXITCODE). Stopping. Tail of log:"
        Get-Content $log -Tail 25
        exit $LASTEXITCODE
    }
    Write-Host "=== [$($s.Name)] done ==="
}

Write-Host "All stages complete. Candidates: outputs/hcdr3_candidates.jsonl"
Write-Host "In the morning: grep 'kd_unit_sanity' / 'strong=' in logs/${ts}_preprocess_asd.log to confirm the KD fix fired."
