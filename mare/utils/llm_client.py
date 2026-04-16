"""
remarl/mare/utils/llm_client.py
--------------------------------
Thin LLM client that wraps Ollama (local), OpenAI, and Anthropic.
MARE agents call this — it is not RL-specific.

Default provider is "ollama" which requires Ollama running locally at
http://localhost:11434  (run: `ollama serve`)
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Simple LLM wrapper used by MARE agents.

    Args:
        provider:    "ollama" | "openai" | "anthropic"
        model:       model string e.g. "llama3.1:8b", "gemma4:latest"
        temperature: generation temperature
        max_tokens:  max output tokens
        base_url:    Ollama server URL (only used when provider="ollama")
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        base_url: str = "http://localhost:11434",
    ):
        self.provider    = provider
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.base_url    = base_url.rstrip("/")
        self._client     = None

    # ── Public API ────────────────────────────────────────────────────────

    def call(self, prompt: str, system_prompt: str = "") -> str:
        """Send a prompt and return the response text."""
        if self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        else:
            raise ValueError(
                f"Unknown provider: '{self.provider}'. "
                "Use 'ollama', 'openai', or 'anthropic'."
            )

    # ── Ollama ────────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call a local Ollama model via the /api/chat endpoint."""
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama at "
                f"{self.base_url}.\n"
                "  → Start Ollama with:  ollama serve\n"
                f"  → Pull the model with: ollama pull {self.model}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}\nResponse: {response.text}")

    # ── OpenAI (fallback) ─────────────────────────────────────────────────

    def _call_openai(self, prompt: str, system_prompt: str = "") -> str:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ── Anthropic (fallback) ──────────────────────────────────────────────

    def _call_anthropic(self, prompt: str, system_prompt: str = "") -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()

    # ── Health check ──────────────────────────────────────────────────────

    @staticmethod
    def check_ollama_health(base_url: str = "http://localhost:11434") -> bool:
        """Return True if Ollama is reachable. Prints a helpful error if not."""
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.error(
                "Ollama is not running. Start it with:  ollama serve"
            )
            return False

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        llm_cfg = config.get("llm", {})
        return cls(
            provider=llm_cfg.get("provider", "ollama"),
            model=llm_cfg.get("model", "llama3.1:8b"),
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 2048),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        )

    @classmethod
    def for_agent(cls, agent_role: str, config: dict) -> "LLMClient":
        """Create an LLMClient using the per-agent model from config."""
        llm_cfg = config.get("llm", {})
        agent_models = llm_cfg.get("agent_models", {})
        model = agent_models.get(agent_role, llm_cfg.get("model", "llama3.1:8b"))
        return cls(
            provider=llm_cfg.get("provider", "ollama"),
            model=model,
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 2048),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        )
