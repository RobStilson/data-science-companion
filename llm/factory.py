import os

# This module is the single place in the codebase that knows which LLM provider
# is being used. Every node that needs an LLM calls get_llm() — they never
# instantiate a model themselves. That way switching providers (Anthropic →
# OpenAI → Ollama) only requires changing the .env file, not the node code.

# Maps provider names to the environment variable that holds their API key.
# Ollama runs locally so it has no key requirement.
_REQUIRED_KEY: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_llm():
    # Read the provider and model from environment variables so they can be
    # changed without touching source code.
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    # Walrus operator (:=) assigns and tests in one expression.
    # If the provider needs a key AND that key is missing, raise early with
    # a clear message rather than getting a cryptic HTTP 401 later.
    if key_var := _REQUIRED_KEY.get(provider):
        if not os.getenv(key_var):
            raise EnvironmentError(f"{key_var} is not set. Add it to your .env file.")

    # Lazy imports — only the chosen provider's package is imported at runtime.
    # This means you don't need langchain-openai installed if you're using Anthropic.
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model)
    elif provider == "ollama":
        # Ollama is a local inference server — no API key needed.
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Choose anthropic, openai, or ollama."
        )
