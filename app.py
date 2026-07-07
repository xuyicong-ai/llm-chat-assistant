import streamlit as st
from llm import ask_llm


st.title("我的第一个AI助手")


question = st.text_input("请输入问题")


if question:

    answer = ask_llm(question)

    st.write(answer)