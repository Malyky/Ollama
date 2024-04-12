import streamlit as st

st.title('Hello, Conny!')
st.write('Ich liebe dich über alles')
st.write('Über alles')

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain_community.llms import Ollama
import streamlit as st
from apikey import apikey
import os
import tempfile
from streamlit_option_menu import option_menu
from typing import Iterable
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableGenerator

#with st.sidebar:
#    selected = option_menu("Main Menu", ["ChatGeneration", 'SummarizePdfs', 'SummarizeWordFiles'],
#                           icons=['house', 'gear'], menu_icon="cast", default_index=1)
#    selected

class Summary:
    def __init__(self, name, text):
        self.name = name
        self.text = text
from menu import menu_with_redirect

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

# Set up OpenAI API
#os.environ["OPENAI_API_KEY"] = apikey
llm = Ollama(model="llama2",
             temperature=0.9,
             # Number of Layers to send to the the gpu
             num_gpu=32,
             )


def parse(ai_message: dict) -> str:
    """Parse the AI message."""
    return ai_message["output_text"]



def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
    for chunk in chunks:
        yield chunk.content.swapcase()


streaming_parse = RunnableGenerator(streaming_parse)

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())

        loader: PyPDFLoader = PyPDFLoader(temp_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        summaries.append((summary, pdf_file.name))

        # Delete the temporary file
        os.remove(temp_path)

    return summaries

def summarizeWordFiles(wordFiles):
    summaries = []
    for wordFile in wordFiles:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(wordFile.read())

        loader: Docx2txtLoader = Docx2txtLoader(temp_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce") | parse
        summary = chain.invoke(docs)
        summaries.append(Summary(wordFile.name, summary))

        # Delete the temporary file
        os.remove(temp_path)

    return summaries

# Streamlit App
st.title("Multiple Documents Summarizer")

# Allow user to upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
wordFiles = st.file_uploader("Upload Word files", type="docx", accept_multiple_files=True)

if pdf_files:
    # Generate summaries when the "Generate Summary" button is clicked
    if st.button("Generate Summary"):
        st.subheader("Summaries:")
        summaries = summarize_pdfs_from_folder(pdf_files)
        for i, summary in enumerate(summaries):
            st.write(f"Summary for PDF {i+1}:")
            st.write(summary)
if wordFiles:
    # Generate summaries when the "Generate Summary" button is clicked
    if st.button("Generate Summary"):
        st.subheader("Summaries:")
        summaries = summarizeWordFiles(wordFiles)
        for i, summary in enumerate(summaries):
            st.write(f"Summary for{summary.name} :")
            st.write(summary.text)

