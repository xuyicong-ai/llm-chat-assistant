import os
from openai import OpenAI
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT, build_rag_prompt

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def get_llm_response(messages):
    """
    messages: list[dict]
    例如：
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"}
    ]
    """

    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + messages

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=final_messages,
        temperature=0.7,
        max_tokens=500
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
        cache_hit_tokens / 1_000_000 * 0.02
        + cache_miss_tokens / 1_000_000 * 1.0
        + usage.completion_tokens / 1_000_000 * 2.0
    )

    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost": estimated_cost
        }
    }


def get_rag_response(messages, context):
    question = messages[-1]["content"]
    rag_prompt = build_rag_prompt(question, context)

    rag_messages = messages[:-1] + [
        {
            "role": "user",
            "content": rag_prompt
        }
    ]

    return get_llm_response(rag_messages)