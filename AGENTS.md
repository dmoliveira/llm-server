# llm-server contributor guide

- Use `uv` for Python dependencies and tooling.
- Keep public commands centralized in `Makefile`; every user-facing target must appear in `make help`.
- Run `make check` before opening a pull request.
- Never commit Hugging Face cache data, model files, runtime state, logs, or secrets.
- Use `python -m llm_server` for application entry points.
