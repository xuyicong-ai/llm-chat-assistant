from openai import OpenAI

client = OpenAI(
    api_key="sk-6fe0d56fc13241b58007b7fbc0976549",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "用一句话解释机器学习"}
    ]
)

print(response.choices[0].message.content)