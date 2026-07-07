import streamlit as st
from openai import OpenAI

client = OpenAI(
    api_key="sk-6fe0d56fc13241b58007b7fbc0976549",
    base_url="https://api.deepseek.com"
)

st.title("我的第一个AI助手")

question = st.text_input("请输入问题")

if question:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content

    st.write(answer)