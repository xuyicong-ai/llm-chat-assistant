import os
from openai import OpenAI
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT

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
        temperature=0.7
    )

    return response.choices[0].message.content