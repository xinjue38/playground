import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader, CSVLoader
import os
import time

# 0. User information
DEBUG = True
print("""Things to do:
         1. 換成 st.tabs 
         2. 弄chat with data
         3. 弄Personal Workspaces
         4. review code and 整理""")
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

## 3.1 Usage Report
if "total_token_used" not in st.session_state.keys():
    st.session_state.total_token_used = 0
def report_usage():
    usage_col = st.columns([0.3,0.4, 0.3])
    with usage_col[0]:
        st.text(f"Credit Left:")
        
    with usage_col[2]:
        st.text(f"Total token used: {st.session_state.total_token_used}")


## 3.2 Part 1 - Upload File
def upload_file_page():
    report_usage()
    
    ### 3.1.1 Chabot information
    with st.sidebar:
        st.radio("Choose your inference methos",
                ("Use your own API key", "Use platform API key (need paid)"), key="selection")
        
        if st.session_state.selection == "Use your own API key":
            st.session_state.openai_api = ""
            st.text_input('Enter OpenAI API key:', 
                        type='password', 
                        key="openai_api",
                        max_chars=200)
            
            if not st.session_state.openai_api:
                st.warning('Please enter your credentials!', icon='⚠️')
                DISABLE_CHAT = True
            elif not (st.session_state.openai_api.startswith('sk-') and len(st.session_state.openai_api) >= 51):
                st.warning('Please enter in correct format!', icon='⚠️')
                DISABLE_CHAT = True
            else:
                st.success(' Enjoy your journey!', icon='👉')
                DISABLE_CHAT = False
            
        elif st.session_state.selection == "Use platform API key (need paid)":
            st.session_state.openai_api = st.secrets["OPENAI_API_KEY"]
            st.success(' Enjoy your journey!', icon='👉')
            DISABLE_CHAT = False
        
        selected_model = st.selectbox('Choose your ChatGPT', ['GPT-3.5', 'GPT-4'], key='selected_model')
        if selected_model == 'GPT-3.5':
            st.session_state.model = "gpt-3.5-turbo"
            st.session_state.MAX_TOKEN = 4096 // 2 
        elif selected_model == 'GPT-4':
            st.session_state.model = "gpt-4"
            st.session_state.MAX_TOKEN = 8192 // 2
        
        expander = st.expander("Advanced Settings")
        expander.slider('temperature', min_value=0.0,  max_value=2.0, value=0.7, step=0.01, key="temperature")
        expander.slider('max_tokens', min_value=128,   max_value=st.session_state.MAX_TOKEN, value=512, step=64, key="max_length")
        expander.markdown('📖 [Learn more about parameter](https://platform.openai.com/docs/api-reference/chat/create)')
        expander.markdown("🔑 [Way to get OpenAI API key](https://platform.openai.com/account/api-keys)")
     
    ### 3.1.2 Main Content for Upload file
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
                                            accept_multiple_files=True,
                                            disabled = DISABLE_CHAT)
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
        st.session_state.list_contents.extend(loader.load())
        # pages = loader.load()
        # st.write(pages[0])
        
        # Remove the file locally after get the information
        os.remove(file_name)

    if len(list_uploaded_files) > 0:
        st.slider('Max number of words in summary', min_value=10,  max_value=300, value=50, step=1, key="max_summarize_text")
        st.button("👉 Start Chat with your data!", key="start_chat")
        

## 3.3 Part 2 - Summarization
def display_summarization():
    query_params = st.experimental_get_query_params()
    
    
    if query_params["user"][0] == USER_ID:
        with st.spinner("Summarizing..."):
            
            if not DEBUG:
                from langchain.chains.mapreduce import MapReduceChain
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
                from langchain.chat_models import ChatOpenAI
                from langchain.chains.llm import LLMChain
                from langchain.prompts import PromptTemplate
                from langchain.chains.combine_documents.stuff import StuffDocumentsChain
                from langchain.callbacks import get_openai_callback
                
                # Map reduce method (from https://python.langchain.com/docs/use_cases/summarization)
                ## 1. Create an LLM
                llm = llm = ChatOpenAI(model_name = st.session_state.model,
                                       temperature= st.session_state.temperature, 
                                       max_tokens = st.session_state.max_length,    
                                       openai_api_key=st.session_state.openai_api)
                
                ## 2. Map Chain
                map_template = """The following is a set of documents
                                {docs}
                                Based on this list of docs, please identify the main themes 
                                Helpful Answer:"""
                map_prompt = PromptTemplate.from_template(map_template)
                map_chain = LLMChain(llm=llm, prompt=map_prompt)
                
                ## 3. Reduce Chain & Combine Documents Chain
                ### Takes a list of documents, combines them into a single string, and passes this to an LLMChain
                reduce_template = """The following is set of summaries:
                {doc_summaries}
                Take these and distill it into a final, consolidated summary of the main themes""" + f"with at most {st.session_state.max_summarize_text} words.\n" + "Helpful Answer:"
                reduce_prompt = PromptTemplate.from_template(reduce_template)
                reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
                combine_documents_chain = StuffDocumentsChain(
                    llm_chain=reduce_chain, document_variable_name="doc_summaries"
                )
                ### Combines and iteravely reduces the mapped documents
                reduce_documents_chain = ReduceDocumentsChain(
                    combine_documents_chain=combine_documents_chain,  # This is final chain that is called.
                    collapse_documents_chain=combine_documents_chain, # If documents exceed context for `StuffDocumentsChain`
                    token_max=4000,                                   # The maximum number of tokens to group documents into.
                )
                
                ## 4. Combining documents by mapping a chain over them, then combining results
                map_reduce_chain = MapReduceDocumentsChain(
                    llm_chain=map_chain,                           # Map chain
                    reduce_documents_chain=reduce_documents_chain, # Reduce chain
                    document_variable_name="docs",                 # The variable name in the llm_chain to put the documents in
                    return_intermediate_steps=False,                # Return the results of the map steps in the output
                )

                text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=0, 
                                separators=["\n\n", "\n", " ", ""])
                
                split_docs = text_splitter.split_documents(st.session_state.list_contents)

                with get_openai_callback() as cb:
                    response = map_reduce_chain.run(split_docs)
                    st.session_state.total_token_used += cb.total_tokens
                report_usage()
                st.title('Summary:')
                st.write(response)
            else:
                st.write("DEBUGGING...")
        
        if st.button("👉 Summarize another set of documents"):
            st.session_state.start_chat = False
            st.experimental_set_query_params()
            
            
    # Error handling, seem like never happen
    else:
        for secs in range(5,0,-1):
            st.warning("Somethings when wrong ..., it will return to original pages in {secs} s")
            time.sleep(1)
            st.session_state.start_chat = False


## 3.4 Display Part 1 OR Part 2
if not st.session_state.start_chat:
    upload_file_page()
else:
    display_summarization()
    
        
