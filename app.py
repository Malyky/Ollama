import streamlit as st
from menu import menu

# Selectbox to choose role
st.title("Welcome to Llama 2.0")
st.image("llama2picture.jpg", width=500)
menu() # Render the dynamic menu!
