import os


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

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
