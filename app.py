
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(page_title="北海学園大学 法学部FAQチャット", layout="wide")
st.title("北海学園大学 法学部FAQチャット")
st.write("このチャットでは、大学の公式資料に基づいて、法学部の履修に関する質問にお答えします。\n\n※丁寧語（です・ます調）で返答します。公序良俗に反する質問や個人情報には対応していません。")

if "qa" not in st.session_state:
    with st.spinner("資料を読み込んでいます..."):
        pdf_files = ["hougaku_guide_2025_shin.pdf", "hougaku_rule (1).pdf"]
        docs = []
        for file in pdf_files:
            loader = PyPDFLoader(file)
            docs.extend(loader.load())
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
        st.session_state.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            retriever=vectordb.as_retriever()
        )

query = st.text_input("質問を入力してください（例：卒業単位はいくつですか？）")
if query:
    with st.spinner("回答を生成しています..."):
        answer = st.session_state.qa.run(query)
        st.markdown(f"**回答：**\n{answer}")
