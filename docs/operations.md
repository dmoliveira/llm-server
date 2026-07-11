# Operations guide

## State machine

`starting → ready → stopping → stopped` is the normal path. A process that exits during readiness becomes `failed`; an externally terminated process is reconciled to `stopped` on the next status request. State is stored locally in `~/.local/share/llm-server/services.json`, guarded by a lock and written through an atomic rename.

## Multiple models

Run one model per port. This is intentional: MLX-LM has an isolated memory/failure domain per model and client configuration is obvious.

```bash
make start MODEL=qwen3-8b SERVICE=writer MODEL_PORT=8080
make start MODEL=deepseek-r1-8b SERVICE=reasoner MODEL_PORT=8081
make status
```

The controller and models bind to `127.0.0.1`. Do not expose either directly to a network. Use an authenticated TLS reverse proxy only after reviewing its access policy.

## Debugging

| Symptom | Check | Resolution |
| --- | --- | --- |
| `FAILED` after start | `make logs SERVICE=name` | Confirm `make install-mlx`, model access, and available unified memory. |
| Port rejected | `make status` | Pick another `MODEL_PORT` or stop the existing service. |
| Cache missing | `make models` | Prefetch with `make models-download MODEL=…`; MLX-LM also downloads on first start. |
| Service died | `make status` then logs | State reconciles on read; restart after fixing the logged error. |

## Trust boundaries

The caller controls model IDs. Only download models you trust; examine model cards and pin revisions for deployments. Hugging Face tokens are read by Hugging Face's standard environment configuration and are never accepted by this API, persisted in service state, or printed by LLM Server.
