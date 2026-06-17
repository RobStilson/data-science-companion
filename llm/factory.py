import os


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model)
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Choose anthropic, openai, or ollama."
        )
