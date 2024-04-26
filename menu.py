import streamlit as st

def authenticated_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("pages/chatbot.py", label="Chatbot", icon="🤖")
    st.sidebar.page_link("pages/langchain_chatbot.py", label="Langchain Chatbot", icon="🦜")
    st.sidebar.page_link("pages/summarizer.py", label="Summarizer", icon="📝")
    st.sidebar.page_link("pages/agents.py", label="Agents", icon="🕵🏾")
    st.sidebar.page_link("pages/websiteRaq.py", label="WebsiteRaq", icon="🌐")

def menu():
        return authenticated_menu()


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    menu()
