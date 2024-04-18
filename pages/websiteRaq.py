from langchain_chroma import Chroma
import chromadb
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models.ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import bs4
import requests

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.title("Website Retrieval and Question Answering")

# load the document and split it into chunks
loader = WebBaseLoader("")
bs4_strainer = bs4.SoupStrainer(class_=("mw-content-text"))
loader = WebBaseLoader(
    web_paths=("https://de.wikipedia.org/wiki/Deutschland",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(id="mw-content-text")},
)
#loader = UnstructuredPDFLoader("citrus/citrus-reference-3.0.0.pdf", mode="elements", strategy="fast")
#documents = loader.load()

input_text = st.text_input('Enter website to analyze', 'https://www.consol.de/unternehmen/')

#query = st.text_input("What is your question?", "Was sind die werte von consol?")
from typing import Iterable


from langchain_core.runnables import RunnableGenerator


def parse(answer: dict) -> str:
    """Parse the AI message."""
    return answer.get("output_text")
def askQuestion(query):
    loader= WebBaseLoader(input_text)
    documents = loader.load()

    # split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    splittedDocs = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    vectorstore = Chroma.from_documents(splittedDocs, embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

    docs = vectorstore.similarity_search(query)

    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=100)
    llm = ChatOllama(model="llama2")
    chain = load_qa_chain(llm, "stuff") | parse

    result = chain.invoke({"input_documents": docs, "question":query})
    with st.chat_message("assistant"):
        st.write(result)

if prompt := st.chat_input("Ask Question"):
    askQuestion(prompt)

