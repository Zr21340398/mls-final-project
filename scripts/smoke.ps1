param([string]$Python = '.venv/Scripts/python.exe')
$ErrorActionPreference = 'Stop'
function Run-Step([string[]]$StepArgs) {
    & $Python @StepArgs
    if ($LASTEXITCODE -ne 0) { throw "Failed: $StepArgs" }
}
Run-Step @('-m','energyfl','synthetic','--out','data/raw/synthetic.csv')
Run-Step @('-m','energyfl','prepare','--csv','data/raw/synthetic.csv','--out','data/processed/smoke','--start','2013-01-01','--end','2013-01-31','--clients','8')
Run-Step @('-m','energyfl','train','--config','configs/smoke.json','--run','runs/smoke')
Run-Step @('-m','energyfl','test','--run','runs/smoke')
Run-Step @('-m','energyfl','analyze','--runs','runs','--out','reports/smoke')
