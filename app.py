from pathlib import Path

import streamlit as st

from llm import get_rag_response
from rag import (
    build_context,
    build_vectorstore,
    load_pdf_documents,
    load_saved_vectorstore,
    retrieve_from_vectorstore
)


st.set_page_config(
    page_title="LLM Chat Assistant",
    page_icon="🤖"
)

st.title("🤖 LLM Chat Assistant")
st.write("一个支持 PDF 知识库问答的 AI Chat Assistant")


# 缓存加载默认 PDF 向量库
@st.cache_resource
def get_default_vectorstore():
    return load_saved_vectorstore(
        "vectorstores/attention_pdf"
    )


default_vectorstore = get_default_vectorstore()


# 初始化当前知识库
if "active_vectorstore" not in st.session_state:
    st.session_state.active_vectorstore = (
        default_vectorstore
    )

if "active_pdf_name" not in st.session_state:
    st.session_state.active_pdf_name = "test.pdf"


# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []


# 显示当前知识库
st.sidebar.info(
    f"当前知识库："
    f"{st.session_state.active_pdf_name}"
)

# 查找已经保存的上传知识库
saved_index_root = Path(
    "vectorstores/uploads"
)

saved_indexes = {}

if saved_index_root.exists():
    for index_folder in saved_index_root.iterdir():
        if (
            index_folder.is_dir()
            and (index_folder / "index.faiss").exists()
            and (index_folder / "index.pkl").exists()
        ):
            saved_indexes[index_folder.name] = (
                index_folder
            )


# 加载已有知识库
if saved_indexes:
    selected_index_name = st.sidebar.selectbox(
        "选择已保存的知识库",
        options=list(saved_indexes.keys())
    )

    if st.sidebar.button(
        "加载已保存的知识库"
    ):
        with st.spinner(
            "正在加载已有向量数据库..."
        ):
            st.session_state.active_vectorstore = (
                load_saved_vectorstore(
                    str(
                        saved_indexes[
                            selected_index_name
                        ]
                    )
                )
            )

            st.session_state.active_pdf_name = (
                selected_index_name
            )

            st.session_state.messages = []

            st.success(
                f"{selected_index_name} "
                f"知识库加载成功"
            )

# 上传 PDF
uploaded_file = st.sidebar.file_uploader(
    "上传 PDF 知识库",
    type=["pdf"]
)


if uploaded_file is not None:
    file_size_mb = (
        uploaded_file.size / 1024 / 1024
    )

    st.sidebar.success(
        f"已选择：{uploaded_file.name}"
    )

    st.sidebar.caption(
        f"文件大小：{file_size_mb:.2f} MB"
    )

    # 创建上传文件目录
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    # 只保留文件名，防止路径问题
    safe_file_name = Path(
        uploaded_file.name
    ).name

    uploaded_path = (
        upload_dir / safe_file_name
    )

    # 将上传对象保存为本地 PDF
    uploaded_path.write_bytes(
        uploaded_file.getbuffer()
    )

    st.sidebar.caption(
        "PDF 已保存，等待建立知识库"
    )

    # 只有点击按钮后才计算 Embedding
    if st.sidebar.button(
        "建立 PDF 知识库",
        type="primary"
    ):
        with st.spinner(
            "正在解析 PDF 并创建向量数据库..."
        ):
            uploaded_documents = (
                load_pdf_documents(
                    str(uploaded_path)
                )
            )

            # 删除没有正文的页面
            valid_documents = [
                doc
                for doc in uploaded_documents
                if doc.page_content.strip()
            ]

            if not valid_documents:
                st.error(
                    "PDF 没有可提取的文字。"
                    "它可能是扫描件，需要 OCR。"
                )

            else:
                uploaded_vectorstore = (
                    build_vectorstore(
                        valid_documents
                    )
                )

                # 切换到新建的向量库
                st.session_state.active_vectorstore = (
                    uploaded_vectorstore
                )

                st.session_state.active_pdf_name = (
                    safe_file_name
                )

                # 建库完成后立即保存索引，避免刷新后丢失
                index_name = Path(
                    safe_file_name
                ).stem

                index_path = (
                    Path("vectorstores")
                    / "uploads"
                    / index_name
                )

                uploaded_vectorstore.save_local(
                    str(index_path)
                )

                # 切换知识库后清空旧对话
                st.session_state.messages = []

                st.success(
                    f"{safe_file_name} "
                    f"知识库创建并保存成功"
                )

                st.rerun()


# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # 显示历史引用来源
        if message.get("sources"):
            with st.expander("查看引用来源"):
                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):
                    page = source.get("page")

                    if page is not None:
                        source_title = (
                            f"**来源 {index}："
                            f"{source['source']}"
                            f"（第 {page} 页）**"
                        )
                    else:
                        source_title = (
                            f"**来源 {index}："
                            f"{source['source']}**"
                        )

                    st.markdown(source_title)
                    st.write(source["content"])

        # 显示历史 Token 用量
        if message.get("usage"):
            usage = message["usage"]

            st.caption(
                f"本次用量："
                f"输入 {usage['prompt_tokens']} tokens · "
                f"输出 {usage['completion_tokens']} tokens · "
                f"总计 {usage['total_tokens']} tokens · "
                f"估算费用 "
                f"¥{usage['estimated_cost']:.6f}"
            )


# 用户输入
user_input = st.chat_input(
    "请输入你的问题..."
)


if user_input:
    # 保存并显示用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # 检索并生成回答
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            documents = (
                retrieve_from_vectorstore(
                    st.session_state.active_vectorstore,
                    user_input,
                    k=3,
                    max_distance=0.8
                )
            )

            context = build_context(documents)

            if documents:
                result = get_rag_response(
                    st.session_state.messages,
                    context
                )

                assistant_response = (
                    result["content"]
                )

                usage = result["usage"]

            else:
                assistant_response = (
                    "根据当前知识库，"
                    "没有检索到与该问题相关的信息。"
                )

                usage = None

            # 整理引用来源
            sources = []

            for doc in documents:
                page = doc.metadata.get("page")

                sources.append({
                    "source": doc.metadata.get(
                        "source",
                        "未知来源"
                    ),
                    "page": (
                        page + 1
                        if page is not None
                        else None
                    ),
                    "content": doc.page_content
                })

            # 显示回答
            st.write(assistant_response)

            # 显示引用来源
            if sources:
                with st.expander("查看引用来源"):
                    for index, source in enumerate(
                        sources,
                        start=1
                    ):
                        page = source.get("page")

                        if page is not None:
                            source_title = (
                                f"**来源 {index}："
                                f"{source['source']}"
                                f"（第 {page} 页）**"
                            )
                        else:
                            source_title = (
                                f"**来源 {index}："
                                f"{source['source']}**"
                            )

                        st.markdown(source_title)
                        st.write(source["content"])

            # 显示本次 Token 与费用
            if usage:
                st.caption(
                    f"本次用量："
                    f"输入 "
                    f"{usage['prompt_tokens']} tokens · "
                    f"输出 "
                    f"{usage['completion_tokens']} tokens · "
                    f"总计 "
                    f"{usage['total_tokens']} tokens · "
                    f"估算费用 "
                    f"¥{usage['estimated_cost']:.6f}"
                )

    # 保存 AI 回答
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response,
        "sources": sources,
        "usage": usage
    })

# 保存当前知识库索引
if st.sidebar.button("保存当前知识库索引"):
    index_name = Path(
        st.session_state.active_pdf_name
    ).stem

    index_path = (
        Path("vectorstores")
        / "uploads"
        / index_name
    )

    st.session_state.active_vectorstore.save_local(
        str(index_path)
    )

    st.sidebar.success(
        f"索引已保存：{index_name}"
    )
    
# 清空对话
if st.sidebar.button("清空对话"):
    st.session_state.messages = []
    st.rerun()
