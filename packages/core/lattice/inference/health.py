"""Provider reachability checks and user-facing provider error messages.

The API, CLI and evaluation runs share these so a missing local model produces
one actionable message instead of a raw transport error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lattice.config import InferenceProvider, get_settings

_PROBE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class ProviderHealth:
    name: str
    model: str
    reachable: bool
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderUnavailableError(RuntimeError):
    """The configured model cannot be reached; the message is safe to show users."""


def ollama_unreachable_message(base_url: str, model: str) -> str:
    return (
        f"Ollama isn't reachable at {base_url}. "
        f"Start it with `ollama serve`, then `ollama pull {model}`."
    )


def _model_matches(model: str, tags: list[str]) -> bool:
    wanted = model if ":" in model else f"{model}:latest"
    return model in tags or wanted in tags


async def check_provider(timeout: float = _PROBE_TIMEOUT_SECONDS) -> ProviderHealth:
    """Probe the configured provider without generating anything."""
    settings = get_settings()
    provider = settings.default_provider

    if provider == InferenceProvider.OLLAMA:
        import httpx

        base_url = settings.ollama_base_url.rstrip("/")
        model = settings.ollama_model
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                tags = [m.get("name", "") for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            detail = ollama_unreachable_message(base_url, model)
            return ProviderHealth("ollama", model, False, detail)
        if not _model_matches(model, tags):
            detail = (
                f"Ollama is running but model `{model}` is not installed. "
                f"Run `ollama pull {model}`."
            )
            return ProviderHealth("ollama", model, False, detail)
        return ProviderHealth("ollama", model, True)

    if provider == InferenceProvider.ANTHROPIC:
        if settings.anthropic_api_key:
            return ProviderHealth("anthropic", settings.anthropic_model, True)
        return ProviderHealth(
            "anthropic", settings.anthropic_model, False,
            "Set LATTICE_ANTHROPIC_API_KEY to use the Anthropic provider.",
        )

    if provider == InferenceProvider.OPENAI:
        if settings.openai_api_key:
            return ProviderHealth("openai", settings.openai_model, True)
        return ProviderHealth(
            "openai", settings.openai_model, False,
            "Set LATTICE_OPENAI_API_KEY to use the OpenAI provider.",
        )

    return ProviderHealth(str(provider), "", False, f"Unknown provider: {provider}")


def friendly_provider_error(exc: BaseException) -> str | None:
    """Actionable message if `exc` means the model is unreachable, else None."""
    if isinstance(exc, ProviderUnavailableError):
        return str(exc)
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return None
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        settings = get_settings()
        if settings.default_provider == InferenceProvider.OLLAMA:
            return ollama_unreachable_message(settings.ollama_base_url, settings.ollama_model)
        name = settings.default_provider.value
        return f"The {name} API could not be reached. Check your network."
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        settings = get_settings()
        if settings.default_provider == InferenceProvider.OLLAMA:
            model = settings.ollama_model
            return f"Ollama doesn't have model `{model}`. Run `ollama pull {model}`."
    return None
