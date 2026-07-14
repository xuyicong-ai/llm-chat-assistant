from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import get_rag_response


# 读取文档
loader = TextLoader(
    "knowledge/ai_notes.txt",
    encoding="utf-8"
)

documents = loader.load()


# 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

print("chunk数量:", len(chunks))


# Qwen3 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="./models/qwen3-embedding-0.6b",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# FAISS
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS数据库创建成功")

def retrieve_documents(query, k=2, max_distance=0.8):
    scored_results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    documents = []

    for doc, score in scored_results:
        if score <= max_distance:
            documents.append(doc)

    return documents

def retrieve_documents_with_scores(query, k=3):
    return vectorstore.similarity_search_with_score(
        query,
        k=k
    )

def build_context(documents):
    context = "\n\n".join(
        doc.page_content for doc in documents
    )
    return context

# 测试检索
if __name__ == "__main__":
    # 仅在直接运行 rag.py 时执行测试
    query = "Python的装饰器是什么？"

    results = retrieve_documents(query, k=2)
    context = build_context(results)

    test_messages = [
        {
            "role": "user",
            "content": query
        }
    ]

    if results:
        answer = get_rag_response(test_messages, context)
        print("\nDeepSeek RAG 回答:")
        print(answer)
    else:
        print("\n知识库没有检索到相关内容。")

    scored_results = retrieve_documents_with_scores(query, k=3)

    print("\n检索距离:")
    for doc, score in scored_results:
        print("----------------")
        print("距离:", score)
        print("内容:", doc.page_content)