.DEFAULT_GOAL := help
HOST ?= 127.0.0.1
PORT ?= 8787
MODEL ?= qwen3-8b
SERVICE ?= $(MODEL)
MODEL_PORT ?= 8080
LIMIT ?= 10

# Preserve caller input as raw exported values. Recipes consume shell variables so Make never
# expands a user-supplied value as Make syntax before the Python CLI validates it.
export LLM_SERVER_HOST := $(value HOST)
export LLM_SERVER_PORT := $(value PORT)
export LLM_SERVER_MODEL := $(value MODEL)
export LLM_SERVER_SERVICE := $(value SERVICE)
export LLM_SERVER_MODEL_PORT := $(value MODEL_PORT)
export LLM_SERVER_QUERY := $(value QUERY)
export LLM_SERVER_LIMIT := $(value LIMIT)

##@ Help
.PHONY: help
help: ## Show this help, grouped by workflow.
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1;36mLLM Server — Mac-first MLX control plane\033[0m\n\nUsage: make <target> [VARIABLE=value]\n"} /^[a-zA-Z0-9_.-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1;33m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup
.PHONY: install install-mlx lock upgrade
install: ## Install the control plane and developer tooling with uv.
	uv sync --all-extras
install-mlx: ## Install MLX-LM into the uv environment (Apple Silicon only).
	uv sync --extra mlx
lock: ## Resolve the audited dependency declarations without upgrading.
	uv lock
upgrade: ## Upgrade locked dependencies.
	uv lock --upgrade && uv sync --all-extras

##@ Control plane
.PHONY: serve dev
serve: ## Start the local control-plane API (HOST=127.0.0.1 PORT=8787).
	uv run python -m llm_server serve --host "$${LLM_SERVER_HOST}" --port "$${LLM_SERVER_PORT}"
dev: ## Start the API with auto-reload for development.
	uv run uvicorn llm_server.api:app --host 127.0.0.1 --port "$${LLM_SERVER_PORT}" --reload

##@ Models
.PHONY: models models-search models-download models-delete
models: ## List curated aliases and downloaded cache entries.
	uv run python -m llm_server models list
models-search: ## Search Hugging Face (QUERY="qwen mlx" LIMIT=10).
	uv run python -m llm_server models search "$${LLM_SERVER_QUERY}" --limit "$${LLM_SERVER_LIMIT}"
models-download: ## Download MODEL alias/repository (MODEL=qwen3-8b).
	uv run python -m llm_server models download "$${LLM_SERVER_MODEL}"
models-delete: ## Delete cached MODEL alias/repository (MODEL=qwen3-8b).
	uv run python -m llm_server models delete "$${LLM_SERVER_MODEL}"

##@ Services
.PHONY: start stop restart status logs
start: ## Start MODEL as SERVICE on MODEL_PORT (MODEL=qwen3-8b MODEL_PORT=8080).
	uv run python -m llm_server services start "$${LLM_SERVER_MODEL}" --name "$${LLM_SERVER_SERVICE}" --port "$${LLM_SERVER_MODEL_PORT}"
stop: ## Stop SERVICE cleanly (SERVICE=qwen3-8b).
	uv run python -m llm_server services stop "$${LLM_SERVER_SERVICE}"
restart: ## Restart SERVICE with its saved settings.
	uv run python -m llm_server services restart "$${LLM_SERVER_SERVICE}"
status: ## Show every managed service and health state.
	uv run python -m llm_server services status
logs: ## Show the last 80 lines for SERVICE (SERVICE=qwen3-8b).
	uv run python -m llm_server services logs "$${LLM_SERVER_SERVICE}" --lines 80

##@ Quality
.PHONY: format lint test check
format: ## Format and apply safe lint fixes.
	uv run ruff format . && uv run ruff check --fix .
lint: ## Check formatting and lint rules.
	uv run ruff format --check . && uv run ruff check .
test: ## Run the fast, offline test suite.
	uv run pytest -q
check: lint test ## Run the required pre-PR quality gate.
