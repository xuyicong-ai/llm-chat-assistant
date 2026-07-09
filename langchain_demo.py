from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业、耐心的 AI 学习助手。"),
    ("human", "{question}")
])

formatted_prompt = prompt.invoke({
    "question": "什么是 Conversation Memory？"
})

print(formatted_prompt)