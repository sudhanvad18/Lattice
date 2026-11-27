import httpx
import pytest

from lattice.inference import health
from lattice.inference.health import check_provider, friendly_provider_error


class _FakeClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        if self._error:
            raise self._error
        return self._response


def _patch_client(monkeypatch, **kwargs):
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout: _FakeClient(**kwargs))


def _tags(*names):
    req = httpx.Request("GET", "http://x/api/tags")
    return httpx.Response(200, json={"models": [{"name": n} for n in names]}, request=req)


@pytest.fixture(autouse=True)
def ollama_settings(monkeypatch):
    monkeypatch.setenv("LATTICE_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("LATTICE_OLLAMA_MODEL", "llama3.2")


async def test_ollama_down_is_unreachable_with_hint(monkeypatch):
    _patch_client(monkeypatch, error=httpx.ConnectError("All connection attempts failed"))
    result = await check_provider()
    assert not result.reachable
    assert "ollama serve" in result.detail


async def test_ollama_missing_model(monkeypatch):
    _patch_client(monkeypatch, response=_tags("phi3:latest"))
    result = await check_provider()
    assert not result.reachable
    assert "ollama pull llama3.2" in result.detail


async def test_ollama_ready(monkeypatch):
    _patch_client(monkeypatch, response=_tags("llama3.2:latest"))
    assert (await check_provider()).reachable


def test_friendly_error_maps_connect_error():
    message = friendly_provider_error(httpx.ConnectError("All connection attempts failed"))
    assert message and "ollama serve" in message
    assert friendly_provider_error(ValueError("boom")) is None
    assert friendly_provider_error(health.ProviderUnavailableError("down")) == "down"


def test_blank_timeout_has_actionable_message():
    message = friendly_provider_error(httpx.ReadTimeout(""))
    assert "timed out" in message and "retry" in message
