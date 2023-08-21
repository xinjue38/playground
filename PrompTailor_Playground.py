import streamlit as st
from streamlit_extras.app_logo import add_logo
import numpy as np

import os

# 1. Set page configuration and add logo
st.set_page_config(page_title="Playground",
                   page_icon="square_logo-no-bg.png",
                   layout="wide",
)   
add_logo("Resize_rectangle_logo_150x300.png")


# 2. CSS Style
with open("css_style.txt", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
        

# 4. Add main content
## 4.1 Description
st.title("PrompTailor Playground")
st.markdown("""
<br>
<p style="text-align:justify; font-size: 18px;">PrompTailor Playground is a place for everyone to test and play with the prompt. 
We provide some services using pre-defined prompts and pre-trained LLM like ChatGPT for you to play and learn with the prompt. 
</p><br><br>
""", unsafe_allow_html=True)

## 4.2 Button to link to another pages
pages = ["AI_Personal_Tutor", "Docs_Summarize", "Chat_With_Data", "Personal_Workspace"]
emojis = ["👨‍🏫", "📃", "💿", "💰"]
for rows in range(int(np.ceil(len(pages) /3))):
  streamlit_col = st.columns([1/3,1/3,1/3], gap="medium")
  for cols in range(3 if len(pages)-3*rows >= 3 else len(pages)-3*rows):
    with streamlit_col[cols]:
      page = pages[cols+rows*3]
      emoji = emojis[cols+rows*3]
      st.markdown(f"""
            <a href="{page}" target="_self">
            <button class="button">
              <h1 style="font-size:200vm">{emoji}</h1>
              <p style="font-size:30px; font-weight: bold;">{page.replace("_"," ")}</p>
            </button>
            </a>
            """, unsafe_allow_html=True)
  st.write("\n")