from openai import OpenAI
from dotenv import load_dotenv

import streamlit as st
import os
from menu import menu
# Redirect to app.py if not logged in, otherwise show the navigation menu
menu()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
st.title("🦙💬 Ollama Chatbot")
model_options = ["llama3", "gpt-3.5-turbo"]

# Create a box for model configurations
with st.expander("Model Configurations"):
    # Add a slider for temperature
    selected_model = st.selectbox('Choose a model', model_options, key='selected_model')
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, format="%.1f", key='temperature')
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=1.0, step=0.01, key='top_p')
    max_length = st.slider('max_length', min_value=1, max_value=256, value=120, step=8, key='max_length')

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.button('Clear Chat History', on_click=clear_chat_history)

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "llama3"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "How can i help you?"})

base_url = "http://localhost:11434/v1"
if selected_model == "llama3":
    base_url = "http://localhost:11434/v1"
elif selected_model == "gpt-3.5-turbo":
    base_url = "https://api.openai.com/v1"
else:
    base_url = "http://localhost:11434/v1"

client = OpenAI(api_key=api_key, base_url=base_url)

for message in st.session_state.messages:
    if isinstance(message, dict):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        #with st.spinner("Thinking..."):
        stream = client.chat.completions.create(
            model=st.session_state["selected_model"],
            max_tokens=st.session_state["max_length"],
            top_p=st.session_state["top_p"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages if isinstance(m, dict)
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
