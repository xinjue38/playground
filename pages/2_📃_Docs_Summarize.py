import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader, CSVLoader
import os

# 0. User information
USER_ID = "piggy"
FILE_SAVED_DIR = f"tempDir\{USER_ID}"
os.makedirs(FILE_SAVED_DIR, exist_ok=True)

# 1. Set page configuration and add logo
st.set_page_config(page_title="Docs Summarize",
                   page_icon="📃",
                   layout="wide",
)   
add_logo("Resize_rectangle_logo_150x300.png")


# 2. CSS Style 
with open("css_style.txt", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)


# 3. Main content 
if "start_chat" not in st.session_state.keys():
    st.session_state.start_chat = False

if st.session_state.start_chat:
    st.experimental_set_query_params(user=USER_ID)

## 3.1 Part 1 - Upload File
if not st.session_state.start_chat:
    st.experimental_set_query_params()
    MAX_FILE_UPLOADED = 5
    allow_format = ["pdf", "docx", "txt", "pptx", "csv", "xlsx"]
    loader_class_dict = {"pdf":PyPDFLoader, 
                        "docx":UnstructuredWordDocumentLoader,
                        "txt":TextLoader, 
                        "pptx":UnstructuredPowerPointLoader,
                        "csv":CSVLoader,
                        "xlsx":UnstructuredExcelLoader}
    list_uploaded_files = st.file_uploader("Choose maximum 5 file",
                                        type = allow_format ,
                                        accept_multiple_files=True)
    st.session_state.list_contents = []

    if len(list_uploaded_files) > MAX_FILE_UPLOADED:
        st.warning(f"Maximum number of files reached. Only the first {MAX_FILE_UPLOADED} will be processed.")
        list_uploaded_files = list_uploaded_files[:MAX_FILE_UPLOADED]

    for uploaded_file in list_uploaded_files:
        file_name = os.path.join(FILE_SAVED_DIR, uploaded_file.name)
        extension = uploaded_file.name[uploaded_file.name.rindex(".") + 1:]

        # Save the file locally
        with open(file_name,"wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get the dataloade and access content
        loader = loader_class_dict[extension](file_name)
        st.session_state.list_contents.append(loader.load())
        # pages = loader.load()
        # st.write(pages[0])
        
        # Remove the file locally after get the information
        os.remove(file_name)

    if len(list_uploaded_files) > 0:
        st.button("👉 Start Chat with your data!", key="start_chat")
        

## 3.1 Part 2 - Chat with data
else:
    query_params = st.experimental_get_query_params()
    if query_params["user"][0] == USER_ID:
        st.write("Start Chat")
        for pages in st.session_state.list_contents:
            st.write(pages[0])
