import os

_REQUIRED_KEY: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    if key_var := _REQUIRED_KEY.get(provider):
        if not os.getenv(key_var):
            raise EnvironmentError(f"{key_var} is not set. Add it to your .env file.")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Choose anthropic, openai, or ollama."
        )
