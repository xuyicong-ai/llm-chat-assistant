# LLM Chat Assistant

一个基于 DeepSeek、LangChain、FAISS 和 Streamlit 构建的本地 RAG 知识库问答应用。

## 功能

- 支持连续对话
- 支持同时上传多个 PDF
- 将多个 PDF 合并为一个知识库
- 使用本地 Qwen Embedding 模型生成文本向量
- 使用 FAISS 进行相似度检索
- 根据指定的讲稿或论文进行检索
- 显示引用文件、页码和检索距离
- 显示 Token 用量和估算费用
- 支持保存和加载本地知识库索引
- 支持引用来源去重
- 支持文件解析、建库、检索和 API 调用异常处理

## 技术栈

- Python
- Streamlit
- DeepSeek API
- LangChain
- Hugging Face Embeddings
- Qwen3 Embedding
- FAISS
- PyPDF

## 项目结构

```text
llm-chat-assistant/
├── app.py
├── llm.py
├── prompt.py
├── rag.py
├── requirements.txt
├── readme.md
├── knowledge/
├── models/
├── uploads/
└── vectorstores/
```

主要文件：

- `app.py`：Streamlit 页面及交互逻辑
- `llm.py`：DeepSeek API 调用、Token 统计和费用估算
- `prompt.py`：系统提示词和 RAG 提示词
- `rag.py`：PDF 解析、文本切分、Embedding 和 FAISS 检索

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 配置 API Key

在项目根目录创建 `.env` 文件：

```text
DEEPSEEK_API_KEY=你的_API_Key
```

不要将 `.env` 提交到 Git 仓库。

## 配置 Embedding 模型

项目默认从下面的目录加载本地模型：

```text
models/qwen3-embedding-0.6b
```

对应代码位于 `rag.py`：

```python
model_name="./models/qwen3-embedding-0.6b"
```

运行应用前，需要确保该目录中存在完整的 Qwen3 Embedding 模型文件。

## 启动应用

在项目根目录运行：

```powershell
streamlit run app.py
```

然后根据终端提示在浏览器中打开应用。

## 使用方法

1. 在侧边栏上传一个或多个 PDF。
2. 点击“建立 PDF 知识库”。
3. 等待 PDF 解析、文本切分和向量化完成。
4. 在聊天输入框中提出问题。
5. 展开“查看引用来源”，检查引用文件、页码和检索距离。
6. 可以在侧边栏加载以前保存的知识库。

如果问题明确指定“讲稿”或“论文”，应用会优先从对应类型的文件中检索。

## 上传限制

- 一次最多上传 10 个 PDF
- PDF 总大小最多为 50 MB
- 没有可提取文字的 PDF 会被跳过
- 扫描版 PDF 需要先进行 OCR

## 检索参数

当前默认参数：

```python
k=6
max_distance=1.2
```

`k` 表示最多检索的文本片段数量。

FAISS 检索距离越小，通常表示语义越接近，但距离排序不一定完全等于人工相关性判断，因此回答仍需结合引用正文核验。

## 数据安全

下面的本地文件不会提交到 Git：

- `.env`
- 本地模型
- 上传的 PDF
- FAISS 向量索引
- Python 缓存文件

具体规则见 `.gitignore`。

## 注意事项

- 首次加载本地 Embedding 模型可能需要较长时间。
- 建立大型 PDF 知识库时会占用较多内存和计算资源。
- Token 费用仅为根据当前价格配置计算的估算值。
- 模型回答应以引用资料为依据，重要内容仍需人工核验。