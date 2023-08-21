import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader, CSVLoader
import os

# 1. Set page configuration and add logo
st.set_page_config(page_title="Docs Summarize",
                   page_icon="📃",
                   layout="wide",
)   
add_logo("Resize_rectangle_logo_150x300.png")


# 2. CSS Style 
with open("css_style.txt", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# 3. Sidebar


# 4. Main content
allow_format = ["pdf"]
list_uploaded_files = st.file_uploader("Choose a PDF file",
                                       type = allow_format ,
                                       accept_multiple_files=True)
list_loaders = []
for uploaded_file in list_uploaded_files:
    file_name = uploaded_file.name
    extension = uploaded_file.type
    st.write("filename:", file_name, extension)
    
    # Save file
    with open(file_name) as f: 
      f.write(file_name.getbuffer())    


    if extension == "pdf":     # PDF File
        loader = PyPDFLoader(file_name)
        
    # # Access content
    # pages = loader.load()
    # st.write(pages[0])
    # list_loaders.append(loader)