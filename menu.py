import streamlit as st


def authenticated_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages/summarizer.py", label="Summarizer")
    st.sidebar.page_link("pages/user.py", label="TO BE DONE")


def menu():
        return authenticated_menu()


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    menu()
