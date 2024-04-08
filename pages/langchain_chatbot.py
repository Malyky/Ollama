from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain.output_parsers.combining import CombiningOutputParser

import streamlit as st
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
st.title("🦙💬 Ollama Chatbot")
model_options = ["llama2", "gpt-3.5-turbo"]

# Create a box for model configurations
with st.expander("Model Configurations"):
    # Add a slider for temperature
    selected_model = st.selectbox('Choose a model', model_options, key='selected_model')
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, format="%.1f", key='temperature')
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=1.0, step=0.01, key='top_p')
    max_length = st.slider('max_length', min_value=1, max_value=256, value=120, step=8, key='max_length')

def clear_chat_history():
    st.session_state.messages = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
st.button('Clear Chat History', on_click=clear_chat_history)

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "llama2"
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

base_url = "http://localhost:11434/v1"
chat_model = ChatOllama(model="llama2", api_key=api_key, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
if selected_model == "llama2":
    base_url = "http://localhost:11434/v1"
    chat_model: ChatOllama = ChatOllama(model="llama2", api_key=api_key, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])#, baseUrl=base_url)
elif selected_model == "gpt-3.5-turbo":
    base_url = "https://api.openai.com/v1"
    #TODO add other parameters like temperature etc
    chat_model: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=max_length, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])#, baseUrl=base_url)
else:
    base_url = "http://localhost:11434/v1"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}"),
    ]
)
llmChain: LLMChain = LLMChain(
    prompt=prompt,
    llm=chat_model,
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #Probleme mit Memory https://github.com/langchain-ai/langchain/issues/15835
    #memory=memory
)
#chain = llmChain | StrOutputParser()
chat_modelWithExpanderParameters: ChatOllama = ChatOllama(model="llama2", temperature=temperature, top_p=top_p, num_predict=max_length, api_key=api_key, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])#, baseUrl=base_url)
chain = prompt | chat_model | StrOutputParser()
#st.write_stream(chain.stream({"chat_history": [], "question": "hi"}))
client = OpenAI(api_key=api_key, base_url=base_url)

for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append(
        HumanMessage(content=prompt),
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        #with st.spinner("Thinking..."):
        stream = chain.stream({"messages": st.session_state.messages, "question": prompt})
        response = st.write_stream(stream)
    st.session_state.messages.append(
        AIMessage(content=response),
    )
