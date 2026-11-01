# Network tools outside the package

| Tool | What it does |
|---|---|
| `g9_amd_probe.py` | G9's bounded AMD attempt (register G9, M19): the network on a ROCm GPU through AMD's PyTorch for Windows - versions, device checks, GPU-against-CPU numerics, torch.compile, eager evaluations/s, one search through the C API; run with a separate ROCm environment's interpreter, never the project `.venv`; every figure AMD-labelled (`docs/MEASUREMENTS.md`, "G9's bounded AMD attempt") |
