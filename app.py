import streamlit as st
from llm import get_llm_response

st.set_page_config(
    page_title="LLM Chat Assistant",
    page_icon="🤖"
)

st.title("🤖 LLM Chat Assistant")
st.write("一个支持多轮对话的 AI Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    # 保存用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # 调用 LLM
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            assistant_response = get_llm_response(st.session_state.messages)
            st.write(assistant_response)

    # 保存 AI 回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response
    })

# 清空对话按钮
if st.sidebar.button("清空对话"):
    st.session_state.messages = []
    st.rerun()