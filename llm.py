from openai import OpenAI

from config import (
    CACHE_HIT_PRICE,
    CACHE_MISS_PRICE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OUTPUT_TOKEN_PRICE
)
from prompt import SYSTEM_PROMPT, build_rag_prompt


client = (
    OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )
    if DEEPSEEK_API_KEY
    else None
)


def get_llm_response(messages):
    """
    messages: list[dict]

    例如：
    [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好"}
    ]
    """
    if client is None:
        raise RuntimeError(
            "没有找到 DEEPSEEK_API_KEY。"
            "请在 .env 文件中配置 API Key，"
            "然后重新启动应用。"
        )

    final_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ] + messages

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=final_messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )

    usage = response.usage

    cache_hit_tokens = getattr(
        usage,
        "prompt_cache_hit_tokens",
        0
    ) or 0

    cache_miss_tokens = getattr(
        usage,
        "prompt_cache_miss_tokens",
        usage.prompt_tokens - cache_hit_tokens
    ) or 0

    estimated_cost = (
        cache_hit_tokens
        / 1_000_000
        * CACHE_HIT_PRICE
        + cache_miss_tokens
        / 1_000_000
        * CACHE_MISS_PRICE
        + usage.completion_tokens
        / 1_000_000
        * OUTPUT_TOKEN_PRICE
    )

    return {
        "content": (
            response.choices[0].message.content
        ),
        "usage": {
            "prompt_tokens": (
                usage.prompt_tokens
            ),
            "completion_tokens": (
                usage.completion_tokens
            ),
            "total_tokens": (
                usage.total_tokens
            ),
            "estimated_cost": estimated_cost
        }
    }


def get_rag_response(messages, context):
    question = messages[-1]["content"]

    rag_prompt = build_rag_prompt(
        question,
        context
    )

    rag_messages = messages[:-1] + [
        {
            "role": "user",
            "content": rag_prompt
        }
    ]

    return get_llm_response(rag_messages)