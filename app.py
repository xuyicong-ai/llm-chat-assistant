from pathlib import Path
from langchain_core.documents import Document
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
            try:
                selected_path = (
                    saved_indexes[
                        selected_index_name
                    ]
                )

                # 先加载到临时变量
                loaded_vectorstore = (
                    load_saved_vectorstore(
                        str(selected_path)
                    )
                )

                # 加载成功后再切换知识库
                st.session_state.active_vectorstore = (
                    loaded_vectorstore
                )

                st.session_state.active_pdf_name = (
                    selected_index_name
                )

                st.session_state.messages = []

                st.success(
                    f"{selected_index_name} "
                    f"知识库加载成功"
                )

            except Exception as error:
                st.error(
                    f"{selected_index_name} "
                    f"知识库加载失败。"
                )

                with st.expander(
                    "查看加载错误详情"
                ):
                    st.code(str(error))

# 上传多个 PDF
uploaded_files = st.sidebar.file_uploader(
    "上传 PDF 知识库",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    max_pdf_count = 10
    max_total_size_mb = 50

    total_size_mb = sum(
        uploaded_file.size
        for uploaded_file in uploaded_files
    ) / 1024 / 1024

    upload_is_valid = True

    if len(uploaded_files) > max_pdf_count:
        st.sidebar.error(
            f"一次最多上传 "
            f"{max_pdf_count} 个 PDF。"
        )

        upload_is_valid = False

    if total_size_mb > max_total_size_mb:
        st.sidebar.error(
            f"PDF 总大小不能超过 "
            f"{max_total_size_mb} MB。"
        )

        upload_is_valid = False
    st.sidebar.success(
        f"已选择 {len(uploaded_files)} 个 PDF，"
        f"共 {total_size_mb:.2f} MB"
    )

    for uploaded_file in uploaded_files:
        file_size_mb = (
            uploaded_file.size / 1024 / 1024
        )

        st.sidebar.caption(
            f"{uploaded_file.name} · "
            f"{file_size_mb:.2f} MB"
        )
    # 创建上传文件目录
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    st.sidebar.caption(
        "等待建立多 PDF 知识库"
    )

    # 只有点击按钮后，才保存和解析所有 PDF
    if st.sidebar.button(
        "建立 PDF 知识库",
        type="primary",
        disabled=not upload_is_valid
    ):
        with st.spinner(
            "正在解析多个 PDF 并创建向量数据库..."
        ):
            # 用于保存所有 PDF 解析出来的页面
            all_documents = []

            # 用于记录成功加入知识库的文件名
            valid_pdf_names = []

            # 依次处理每一个上传的 PDF
            for uploaded_file in uploaded_files:
                # 清理文件名，避免路径问题
                safe_file_name = Path(
                    uploaded_file.name
                ).name

                try:
                    uploaded_path = (
                        upload_dir / safe_file_name
                    )

                    # 将当前 PDF 保存到 uploads 目录
                    uploaded_path.write_bytes(
                        uploaded_file.getbuffer()
                    )

                    # 解析当前 PDF
                    pdf_documents = (
                        load_pdf_documents(
                            str(uploaded_path)
                        )
                    )

                    # 删除没有正文的页面
                    valid_pages = [
                        doc
                        for doc in pdf_documents
                        if doc.page_content.strip()
                    ]

                    if valid_pages:
                        # 将有效页面加入总文档列表
                        all_documents.extend(
                            valid_pages
                        )

                        valid_pdf_names.append(
                            safe_file_name
                        )

                    else:
                        st.warning(
                            f"{safe_file_name} "
                            f"没有可提取的文字，已跳过。"
                        )

                except Exception as error:
                    st.error(
                        f"{safe_file_name} 处理失败，"
                        f"已跳过该文件。"
                    )

                    # 将详细错误放入折叠区域
                    with st.expander(
                        f"查看 {safe_file_name} 的错误详情"
                    ):
                        st.code(str(error))

            # 所有 PDF 都解析完成后再统一判断
            if not all_documents:
                st.error(
                    "上传的 PDF 都没有可提取的文字。"
                    "它们可能是扫描件，需要 OCR。"
                )

            else:
                try:
                    # 使用所有 PDF 的页面建立向量库
                    uploaded_vectorstore = (
                        build_vectorstore(
                            all_documents
                        )
                    )

                    # 生成知识库显示名称
                    knowledge_base_name = " + ".join(
                        valid_pdf_names
                    )

                    # 生成索引目录名称
                    index_name = "__".join(
                        Path(pdf_name).stem
                        for pdf_name in valid_pdf_names
                    )

                    index_path = (
                        Path("vectorstores")
                        / "uploads"
                        / index_name
                    )

                    # 先保存索引
                    uploaded_vectorstore.save_local(
                        str(index_path)
                    )

                    # 保存成功后再切换当前知识库
                    st.session_state.active_vectorstore = (
                        uploaded_vectorstore
                    )

                    st.session_state.active_pdf_name = (
                        knowledge_base_name
                    )

                    # 切换知识库后清空旧对话
                    st.session_state.messages = []

                    st.success(
                        f"已将 {len(valid_pdf_names)} "
                        f"个 PDF 合并为一个知识库"
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        "向量数据库创建或保存失败，"
                        "当前知识库没有切换。"
                    )

                    with st.expander(
                        "查看建库错误详情"
                    ):
                        st.code(str(error))



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

                    score = source.get("score")

                    if score is not None:
                        st.caption(
                            f"检索距离：{score:.4f}"
                        )

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
            # 先设置默认值，确保失败后变量仍然存在
            documents: list[Document] = []
            usage = None

            try:
                # 从当前知识库检索资料
                documents = (
                    retrieve_from_vectorstore(
                        st.session_state.active_vectorstore,
                        user_input,
                        k=6,
                        max_distance=1.2
                    )
                )

                context = build_context(documents)

                if documents:
                    # 调用模型生成回答
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

            except Exception as error:
                assistant_response = (
                    "回答生成失败，请稍后重试。"
                    "当前对话和知识库没有丢失。"
                )

                st.error(
                    "检索或模型调用失败。"
                )

                with st.expander(
                    "查看问答错误详情"
                ):
                    st.code(str(error))

            # 整理引用来源
            sources = []

            # 记录已经添加过的“文件 + 页码”
            seen_sources = set()

            for doc in documents:
                page = doc.metadata.get("page")

                source_name = doc.metadata.get(
                    "source",
                    "未知来源"
                )

                # 文件路径和页码共同组成唯一标识
                source_key = (
                    source_name,
                    page
                )

                # 相同文件的相同页码只显示一次
                if source_key in seen_sources:
                    continue

                seen_sources.add(source_key)

                sources.append({
                    "source": source_name,
                    "page": (
                        page + 1
                        if page is not None
                        else None
                    ),
                    "score": doc.metadata.get(
                        "distance_score"
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

                        score = source.get("score")

                        if score is not None:
                            st.caption(
                                f"检索距离：{score:.4f}"
                            )

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
    try:
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

    except Exception as error:
        st.sidebar.error(
            "当前知识库索引保存失败。"
        )

        with st.sidebar.expander(
            "查看保存错误详情"
        ):
            st.code(str(error))
    
# 清空对话
if st.sidebar.button("清空对话"):
    st.session_state.messages = []
    st.rerun()
