import streamlit as st
from llm import get_llm_response
from llm import get_rag_response
from rag import retrieve_documents, build_context

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

        if message.get("sources"):
            with st.expander("查看引用来源"):
                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):
                    st.markdown(
                        f"**来源 {index}：{source['source']}**"
                    )
                    st.write(source["content"])

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
            documents = retrieve_documents(user_input, k=1)
            context = build_context(documents)

            if documents:
                assistant_response = get_rag_response(
                    st.session_state.messages,
                    context
                )
            
            else:
                assistant_response = (
                    "根据当前知识库，没有检索到与该问题相关的信息。"
                )
            
            sources = []

            for doc in documents:
                sources.append({
                    "source": doc.metadata.get("source", "未知来源"),
                    "content": doc.page_content
                })
            st.write(assistant_response)
            if sources:
                with st.expander("查看引用来源"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**来源 {index}：{source['source']}**"
                        )
                        st.write(source["content"])

    # 保存 AI 回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response,
        "sources": sources
    })

# 清空对话按钮
if st.sidebar.button("清空对话"):
    st.session_state.messages = []
    st.rerun()