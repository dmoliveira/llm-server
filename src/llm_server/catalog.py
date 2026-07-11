"""Curated MLX-friendly aliases and Hugging Face cache operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download


@dataclass(frozen=True)
class Model:
    alias: str
    repository: str
    provider: str
    family: str
    size: str
    quantization: str
    context: str
    note: str


CATALOG = (
    Model(
        "qwen3-8b",
        "mlx-community/Qwen3-8B-4bit",
        "Alibaba",
        "Qwen 3",
        "8B",
        "4-bit",
        "32K",
        "Strong daily default",
    ),
    Model(
        "qwen3-30b-a3b",
        "mlx-community/Qwen3-30B-A3B-4bit",
        "Alibaba",
        "Qwen 3 MoE",
        "30B/A3B",
        "4-bit",
        "32K",
        "Efficient MoE",
    ),
    Model(
        "gemma3-12b",
        "mlx-community/gemma-3-12b-it-4bit",
        "Google",
        "Gemma 3",
        "12B",
        "4-bit",
        "128K",
        "Multimodal family",
    ),
    Model(
        "deepseek-r1-8b",
        "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        "DeepSeek",
        "R1 Distill",
        "8B",
        "4-bit",
        "32K",
        "Reasoning model",
    ),
    Model(
        "mistral-small-24b",
        "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
        "Mistral",
        "Small",
        "24B",
        "4-bit",
        "32K",
        "Capable local model",
    ),
    Model(
        "glm-4-9b",
        "mlx-community/GLM-4-9B-Chat-4bit",
        "Z.ai",
        "GLM 4",
        "9B",
        "4-bit",
        "128K",
        "Chinese and English",
    ),
    Model(
        "llama-3.2-3b",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "Meta",
        "Llama 3.2",
        "3B",
        "4-bit",
        "128K",
        "Small smoke-test model",
    ),
)


def resolve(identifier: str) -> Model:
    return next(
        (m for m in CATALOG if m.alias == identifier),
        Model(
            identifier.replace("/", "--"),
            identifier,
            "Custom",
            "Custom",
            "—",
            "—",
            "—",
            "Explicit repository",
        ),
    )


def models() -> list[dict[str, str]]:
    return [asdict(m) for m in CATALOG]


def download(identifier: str, revision: str | None = None) -> str:
    return str(snapshot_download(repo_id=resolve(identifier).repository, revision=revision))


def cached_models() -> list[dict[str, Any]]:
    try:
        cache = scan_cache_dir()
    except Exception:
        return []
    return [
        {"repository": r.repo_id, "size_bytes": r.size_on_disk, "revisions": len(r.revisions)}
        for r in cache.repos
    ]


def delete(identifier: str) -> None:
    repository = resolve(identifier).repository
    try:
        cache = scan_cache_dir()
    except Exception as error:
        raise ValueError("No Hugging Face model cache exists yet") from error
    matches = [r for r in cache.repos if r.repo_id == repository]
    if not matches:
        raise ValueError(f"No downloaded cache entry for {repository}")
    cache.delete_revisions(*(v.commit_hash for r in matches for v in r.revisions)).execute()


def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "repository": m.id,
            "downloads": m.downloads or 0,
            "likes": m.likes or 0,
            "updated": str(m.last_modified or ""),
        }
        for m in HfApi().list_models(
            search=query, sort="downloads", direction=-1, limit=min(limit, 50)
        )
    ]
