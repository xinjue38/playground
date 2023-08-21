import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationSummaryBufferMemory

# 1. Set page configuration and add logo
st.set_page_config(page_title="AI Personal Tutor",
                   page_icon="👨‍🏫",
                   layout="wide",
)   
add_logo("Resize_rectangle_logo_150x300.png")


# 2. CSS Style 
with open("css_style.txt", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

## 2.1 Things to do
# st.subheader("Things to do")
# st.text("""
#         1. Personal AI Prompt 弄好 + AI part
#            - 先复习python, 写好, 再看github prompt
#         2. DOc Summarize (Upload File section 弄好 + Prompt)
#         3. Chat with Data (Upload File section 弄好 + Prompt)
#         """)

# 3. Add sidebar content
with st.sidebar:
    selection = st.radio("Choose your inference methos",
                         ("Use your own API key", "Use platform API key (need paid)"))
    
    if selection == "Use your own API key":
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
        
    elif selection == "Use platform API key (need paid)":
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
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    expander = st.expander("Advanced Settings")
    show_template     = expander.checkbox("Show Prompt Template")
    temperature       = expander.slider('temperature', min_value=0.0,  max_value=2.0, value=0.7, step=0.01)
    max_length        = expander.slider('max_tokens', min_value=128,   max_value=st.session_state.MAX_TOKEN, value=512, step=64)
    expander.markdown('📖 [Learn more about parameter](https://platform.openai.com/docs/api-reference/chat/create)')
    expander.markdown("🔑 [Way to get OpenAI API key](https://platform.openai.com/account/api-keys)")

background_image = """
        <style>
        .main    {
             background           : url("https://github.com/xinjue38/repo_for_image/blob/main/square_logo-no-bg-500-500.png?raw=true");
             background-position  : center;
             background-repeat    : no-repeat;
             background-opacity   : 0.3
         }
        </style>
        """
st.markdown(background_image, unsafe_allow_html=True)

    

# 4. Components require to get message (改这里就好，其他地方不用改)
## 4.0 Inital message to send 
INITIAL_MESSAGE = "How may I assist you today?"
MAX_WINDOW_ALLOW = 30

## 4.1 Define an LLM
llm = ChatOpenAI(model_name = st.session_state.model,
                 temperature=temperature, 
                 max_tokens =max_length,    
                 openai_api_key=st.session_state.openai_api)

### 4.2, 4.3, 4.4 only need run for onces as we cannot initialize memory for many time, memory will loss
if "memory" not in st.session_state.keys():
    ## 4.2 Create system prompt and prompt template
    SYSTEM_PROMPT = "You are a good piggy that love to eat strawberry."
    st.session_state.template = SYSTEM_PROMPT + """
    Current conversation:
    {chat_history}
    Human: {human_input} 
    AI:"""

    st.session_state.prompts = PromptTemplate(input_variables=["chat_history", "human_input"], 
                                              template=st.session_state.template)

    ## 4.3 Select a memory type
    st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", max_token_limit=100)

    ## 4.4 Append conversation message (optional)
    st.session_state.memory.save_context({"input": "Who are u?"}, 
                                         {"output": "Oink Oink! I am a piggy who like to eat strawberry"})

## 4.5 Create LLMChain 
conversation = LLMChain(llm=llm, 
                        memory = st.session_state.memory, 
                        prompt = st.session_state.prompts, 
                        verbose= True)

## 4.6 Function to generate response from the prompt
def generate_response(prompt_input):
    return {"text":"Debug"}
    with get_openai_callback() as cb:
        response = conversation(inputs ={"human_input":prompt_input}, return_only_outputs = not show_template)
        st.session_state.total_token_used += cb.total_tokens
    return response


# 5. Chat Bot
## 5.1 Usage report
if "total_token_used" not in st.session_state.keys():
    st.session_state.total_token_used = 0
usage_col = st.columns([0.3,0.4, 0.3])
with usage_col[0]:
    st.text(f"Credit Left:")
    
with usage_col[2]:
    st.text(f"Total token used: {st.session_state.total_token_used}")


## 5.2 Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": INITIAL_MESSAGE}]

## 5.3 Display chat messages recursively (for every run)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            placeholder = st.empty()
            placeholder.markdown(message["content"])
        elif message["role"] == "user":
            for mes in message["content"].split("\n"):
                st.write(mes)

## 5.4 Only allow MAX_WINDOW_ALLOW windows
if len(st.session_state.messages) > MAX_WINDOW_ALLOW*2:
    DISABLE_CHAT = True
    st.warning(f"You have exceed maximum window allow [{MAX_WINDOW_ALLOW}], to continue conversation, please resfresh or open new tab")

## 5.5 Generate a new response if have new input
if prompt := st.chat_input(disabled=DISABLE_CHAT, max_chars = st.session_state.MAX_TOKEN*4):
    if len(st.session_state.messages) <= MAX_WINDOW_ALLOW*2:
        # If not show template, show the input text first
        if not show_template:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.text(prompt)
        
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
        
        # If show template, show the input text after "Thinking"
        if show_template:
            prompt = st.session_state.prompts.format(chat_history=response["chat_history"], human_input=response["human_input"])
            print("Current prompt",prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                for mes in prompt.split("\n"):
                    st.markdown(mes)
                
        response = response["text"]        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
