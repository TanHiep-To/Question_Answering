import joblib
import requests
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from types import SimpleNamespace
from tensorflow.keras.models import load_model
import wikipedia 
import openai
import gradio as gr
import torch
from transformers import pipeline
from transformers import AutoTokenizer
import model
import prediction

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

st.markdown(
    """
    <style>
    .logo {
        text-align: center;
    }
    .title {
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 0.5em;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2em;
        margin-bottom: 1em;
        text-align: center;
    }
    .section h3 {
        margin-bottom: 1em;
        font-size: 1.5em;
        text-align: center;
    }
    .stTextArea textarea {
        margin-bottom: 20px;
    }
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 8px;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Đặt các cột với tỉ lệ phù hợp
col1, col2 = st.columns([2, 4])

# Hiển thị hình ảnh trong cột đầu tiên với CSS lớp "logo"
with col1.container():
    st.image("https://cdn3.iconfinder.com/data/icons/contact-us-set-5/256/29-1024.png", width=100)

# Hiển thị tiêu đề trong cột thứ hai với CSS lớp "title" và "subtitle"
with col2.container():
    st.markdown('<div class="title">Question Answering Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Hôm nay bạn như thế nào?</div>', unsafe_allow_html=True)

# Tạo khoảng trống giữa các phần
st.markdown("<br>", unsafe_allow_html=True)

# Đặt hai cột với tỉ lệ phù hợp
col1, col2 = st.columns([2, 2])

# Hiển thị phần "Question and Answer" trong cột đầu tiên với CSS lớp "section"
with col1:
    st.markdown('<div class="section"><h3>Question and Answer</h3></div>', unsafe_allow_html=True)
    question = st.text_area("Question", value="Type your question here...", height=80)
    answer = st.text_area("Answer", value="The answer will appear here", height=80)
    # Centered button
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button('Submit'):
        st.write("Button clicked!")
    st.markdown('</div>', unsafe_allow_html=True)


# Hiển thị phần "Context" trong cột thứ hai với CSS lớp "section"
with col2:
    st.markdown('<div class="section"><h3>Context</h3></div>', unsafe_allow_html=True)
    context = st.text_area("", value="Type your context here...", height=230)
    uploaded_file = st.file_uploader("**Choose a file**")
    if uploaded_file is not None:
        # Use the file here
        pass
    
# with st.spinner('Loading model... Please wait a minute.'):
#     model = model.load_model('../models/qa_large_model/')

# question = st.text_input("Enter question here.  .....")
# context = ""
# add_context = st.checkbox('Add context')
# if add_context:
#     context = st.text_area("Enter context here......")
# else:
#     keywords = wikipedia.search([question])
    
#     for keyword in keywords:
#         try:
#             context += wikipedia.summary(keyword, sentences=10)
#             break
#         except:
#             pass

# print(context)

# if st.button('Submit'):
#     with torch.inference_mode():
#         question, answer = prediction.run_prediction(model, tokenizer, context, question)
#         st.write(f'Question: {question}')
#         st.write(f'Answer: {answer}')