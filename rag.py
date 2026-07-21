from pathlib import Path
from functools import lru_cache
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    MAX_RETRIEVAL_DISTANCE,
    RETRIEVAL_K
)

# 加载 PDF
def load_pdf_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 取得 PDF 文件名
    source_name = Path(file_path).name

    # 将文件名加入每一页的正文，
    # 使文件名也参与向量检索
    for document in documents:
        # 只有存在正文的页面才添加文件名
        if document.page_content.strip():
            document.page_content = (
                f"文档名称：{source_name}\n\n"
                f"{document.page_content}"
            )

    return documents

# 创建 FAISS 向量数据库
def build_vectorstore(documents):

    chunks = splitter.split_documents(documents)

    print("chunk数量:", len(chunks))

    vectorstore = FAISS.from_documents(
        chunks,
        get_embeddings()
    )
    
    return vectorstore

def load_saved_vectorstore(folder_path):
    vectorstore = FAISS.load_local(
        folder_path,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

    return vectorstore

def retrieve_from_vectorstore(
    vectorstore,
    query,
    k=RETRIEVAL_K,
    max_distance=MAX_RETRIEVAL_DISTANCE
):
    scored_results = (
        vectorstore.similarity_search_with_score(
            query,
            k=k
        )
    )

    retrieved_documents = []

    # 检查用户是否明确指定了文档类型
    mentioned_document_types = set()

    if "讲稿" in query:
        mentioned_document_types.add("讲稿")

    if "论文" in query:
        mentioned_document_types.add("论文")

    for doc, score in scored_results:
        source_path = doc.metadata.get(
            "source",
            ""
        )

        source_name = Path(source_path).stem

        # 用户指定文档后，跳过其他文档
        if mentioned_document_types:
            source_matches = any(
                document_type in source_name
                for document_type
                in mentioned_document_types
            )

            if not source_matches:
                continue

        if score <= max_distance:
            # 把检索距离写入文档元数据
            doc.metadata["distance_score"] = float(
                score
            )

            retrieved_documents.append(doc)

    return retrieved_documents

# 构建提供给 LLM 的参考资料
def build_context(documents):
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    return context


# 初始化文本切分器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


# 第一次使用时才加载 Qwen3 Embedding
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
        show_progress=True
    )

# 第一次使用时才创建原 TXT 知识库
@lru_cache(maxsize=1)
def get_default_text_vectorstore():
    loader = TextLoader(
        "knowledge/ai_notes.txt",
        encoding="utf-8"
    )

    documents = loader.load()
    vectorstore = build_vectorstore(documents)

    print("TXT FAISS数据库创建成功")

    return vectorstore


# 在原 TXT 知识库中检索，并进行距离过滤
def retrieve_documents(
    query,
    k=2,
    max_distance=0.8
):
    vectorstore = get_default_text_vectorstore()

    scored_results = (
        vectorstore.similarity_search_with_score(
            query,
            k=k
        )
    )

    retrieved_documents = []

    for doc, score in scored_results:
        if score <= max_distance:
            retrieved_documents.append(doc)

    return retrieved_documents


# 查看原 TXT 知识库的原始检索距离
def retrieve_documents_with_scores(query, k=3):
    vectorstore = get_default_text_vectorstore()

    return vectorstore.similarity_search_with_score(
        query,
        k=k
    )


# 只有直接运行 rag.py 时，才执行 PDF 检索测试
if __name__ == "__main__":
    print("\n正在加载已有 PDF 向量数据库...")

    pdf_vectorstore = load_saved_vectorstore(
        "vectorstores/attention_pdf"
    )

    print("PDF向量数据库加载成功")

    query = "What is multi-head attention?"

    pdf_results = (
        pdf_vectorstore.similarity_search_with_score(
            query,
            k=3
        )
    )

    print("\nPDF检索问题:", query)
    print("\nPDF检索结果:")

    for doc, score in pdf_results:
        print("----------------")
        print("距离:", score)
        print(
            "PDF页码:",
            doc.metadata["page"] + 1
        )
        print(
            "来源:",
            doc.metadata["source"]
        )
        print(
            "内容:",
            doc.page_content[:300]
        )