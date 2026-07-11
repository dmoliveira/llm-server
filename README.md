<p align="center"><img src="site/hero.svg" alt="LLM Server" width="760"></p>

<p align="center">
  <a href="https://github.com/dmoliveira/llm-server/actions"><img src="https://img.shields.io/github/actions/workflow/status/dmoliveira/llm-server/ci.yml?branch=main&label=checks&style=flat-square" alt="checks"></a>
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black?style=flat-square&logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-7ee787?style=flat-square" alt="MIT"></a>
</p>

> **A friendly local control plane for open language models on Apple Silicon.** Download, run, inspect, and stop multiple MLX-LM servers without losing track of ports, logs, or cached models.

## ✨ What it does

- **One model, one port, zero ambiguity.** Launch any number of isolated `mlx_lm.server` processes on `127.0.0.1`; duplicate managed ports are rejected.
- **Human aliases.** Start with `qwen3-8b`, `gemma3-12b`, `deepseek-r1-8b`, `glm-4-9b`, or an explicit Hugging Face repository.
- **A real operations surface.** Rich terminal tables, an OpenAPI control-plane API, safe bounded log tails, atomic state, and a compact browser dashboard.
- **Cache control.** Discover public Hub models, prefetch one, inspect cached repositories, or delete it cleanly.

## 🚀 Quick start

> Requires macOS on Apple Silicon, Python 3.11+, and [`uv`](https://docs.astral.sh/uv/). MLX-LM is installed only when you ask for it.

```bash
git clone https://github.com/dmoliveira/llm-server.git && cd llm-server
make install
make install-mlx
make help

# Model servers use their own ports and expose MLX-LM's OpenAI-compatible API.
make start MODEL=qwen3-8b SERVICE=qwen MODEL_PORT=8080
make start MODEL=gemma3-12b SERVICE=gemma MODEL_PORT=8081
make status
```

Then call a running model at `http://127.0.0.1:8080/v1/chat/completions`. Start the dashboard at `make serve` and visit `http://127.0.0.1:8787`.

## 🧭 Command map

| Need | Command |
| --- | --- |
| See every command | `make help` |
| Browse aliases + cache | `make models` |
| Search the public Hub | `make models-search QUERY="qwen mlx" LIMIT=10` |
| Download / remove a model | `make models-download MODEL=qwen3-8b` / `make models-delete MODEL=qwen3-8b` |
| Start / stop / restart | `make start …` / `make stop SERVICE=qwen` / `make restart SERVICE=qwen` |
| See health + logs | `make status` / `make logs SERVICE=qwen` |

Every Make target calls `python -m llm_server`; no hidden shell scripts or daemon manager are required.

## 🌐 Control-plane API

The controller deliberately binds to localhost. It does **not** provide remote authentication or TLS in v0.1; use a trusted reverse proxy if you must bridge machines.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | visual dashboard |
| `GET /docs` | interactive OpenAPI docs |
| `GET /api/v1/models/catalog` | curated aliases |
| `GET /api/v1/models/downloaded` | local Hub cache inventory |
| `GET /api/v1/models/search?query=qwen` | bounded public Hub search |
| `GET /api/v1/status` | all services |
| `POST /api/v1/services` | spawn a model service |
| `POST /api/v1/services/{name}/stop` | terminate a process group |
| `GET /api/v1/services/{name}/logs` | bounded, service-owned log tail |

## 🧠 Model notes for M5

Choose an MLX-native 4-bit checkpoint first. Unified memory is shared: each running service owns model weights and KV cache. Begin with one 8B model, verify memory headroom, then add services. `--max-kv-size` (available through `services start --max-kv-size …`) is the primary control for long-context memory pressure.

The bundled aliases are discoverability aids, not a trust guarantee. For an explicit repository, inspect its Model Card, pin a revision for reproducibility, and never put Hugging Face tokens in a command, state file, or log.

## 🛡️ Lifecycle guarantees

State writes are atomic and guarded by an advisory file lock. Child servers use a dedicated process group and stop escalates `SIGTERM → SIGKILL`. Services become **ready** only after MLX-LM responds at `/v1/models`; failed starts preserve a clear error and a log file under `~/.local/share/llm-server/logs/`.

## 🤝 Contributing

```bash
make install
make check
```

Please keep public workflows in `Makefile`, use `uv`, and never commit model weights, Hub cache data, local state, logs, or secrets. See [docs/operations.md](docs/operations.md) for the service model.
