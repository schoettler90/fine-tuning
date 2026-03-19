Project

Repository to load and fine-tune large language models (LLMs) for various applications
that can be later used in Ollama.

## Environment

- Python 3.13 (not 3.14 — ML wheels top out at cp313); venv at `.venv` via `uv`
- Install: `uv sync --extra mps` (Apple Silicon) or `uv sync --extra cuda` (NVIDIA Linux)

## Known constraints

- **Unsloth installs on Mac but MPS is unsupported at runtime** — raises `NotImplementedError` when loading `FastLanguageModel` on Apple Silicon; their Mac docs page is for Unsloth Studio (web UI), not the Python API
- **MPS fine-tuning** uses standard PEFT + TRL (no unsloth)
- **xformers** won't compile on macOS (no OpenMP in Apple clang) — excluded via `override-dependencies` in `pyproject.toml`
- **CUDA torch** has no cp314 wheels — Python 3.13 required