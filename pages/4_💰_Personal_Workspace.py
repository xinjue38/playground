import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo

# 1. Set page configuration and add logo
st.set_page_config(page_title="Personal Workspace",
                   page_icon="💰",
                   layout="wide",
)   
add_logo("Resize_rectangle_logo_150x300.png")


# 2. CSS Style 
with open("css_style.txt", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
