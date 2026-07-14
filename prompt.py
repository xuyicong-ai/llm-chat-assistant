SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer questions clearly and accurately.
"""
def build_rag_prompt(question, context):
    return f"""
请根据下面提供的参考资料回答用户问题。

参考资料：
{context}

用户问题：
{question}

回答要求：
1. 优先依据参考资料回答。
2. 如果参考资料没有相关信息，请明确说明。
3. 不要编造参考资料中不存在的内容。
"""