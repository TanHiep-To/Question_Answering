import joblib
import requests
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from types import SimpleNamespace
import openai
import gradio as gr
import torch
import model
import prediction
import ggsearch as gg
from transformers import pipeline, AutoTokenizer

# test
# question = st.text_input("Enter your question:",key="question")
# if st.button("Search"):
#     results = gg.get_google_results(question)
#     documents = [result['snippet'] for result in results]
#     most_similar_document = gg.find_most_similar_documents(question, documents)
#     st.write(most_similar_document)
#     st.write("The best paragraph:")
#     best_paragraph = gg.the_best_paragraph(question, most_similar_document)
#     st.write(best_paragraph)

# Tải tokenizer và model
if "model" not in st.session_state.keys():
    st.session_state.model = model.load_model('../model/model/')
if "tokenizer" not in st.session_state.keys():
    st.session_state.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    

# Load question-answer pipeline
qa_pipeline = pipeline('question-answering', 
                    model=st.session_state.model,
                    tokenizer=st.session_state.tokenizer)

if "question" not in st.session_state.keys():
    st.session_state.question = ""
if "context" not in st.session_state.keys():
    st.session_state.context = ""
if "checkbox" not in st.session_state.keys():
    st.session_state.checkbox = False
if "answer" not in st.session_state.keys():
    st.session_state.answer = ""

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
    .stCheckbox{
        margin-top: -30px;
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
# col1, col2 = st.columns([3, 5])

# Hiển thị phần "Question and Answer" với CSS lớp "section"
# with col1:
st.markdown('<div class="section"><h3>Question</h3></div>', unsafe_allow_html=True)
st.text("")
question = st.text_area("Question", value=st.session_state.question, placeholder="Type your question here...", height=50)

def checkBoxOnChange():
    st.experimental_rerun()

# Hiển thị phần "Context" với CSS lớp "section"
# with col2:
st.markdown('<div class="section"><h3>Context</h3></div>', unsafe_allow_html=True)
checkBox = st.checkbox(label="Give Context", value=st.session_state.checkbox, help="We will automatically search to find a suitable context for you.", label_visibility="visible", on_change=checkBoxOnChange)

if checkBox:
    context = st.text_area("Context", value=st.session_state.context, placeholder="Give your context here...", height=200)
    st.session_state.context = context
    st.session_state.question = question
else:
    st.session_state.question = question
    st.session_state.answer = ""

st.markdown('<div class="center-button">', unsafe_allow_html=True)

if st.button('Submit'): 
    if checkBox:
        result = qa_pipeline({'question': question, 'context': context}) 
        st.text_area("Answer", value=result['answer'], height=80)
    else:
#        # test
#        document = ""
#        paragraph = ""
#        paragraph = document = context = "Paris (phát âm tiếng Pháp: ​[paʁi] ⓘ) là thủ đô và là thành phố đông dân nhất nước Pháp, cũng là một trong ba thành phố phát triển kinh tế nhanh nhất thế giới cùng Luân Đôn và New York và là một trung tâm hành chính của vùng Île-de-France với dân số ước tính là 2.165.423 người tính đến năm 2019, trên diện tích hơn 105,4 km2 (40,7 dặm vuông Anh)."

        results = gg.get_google_results(question)
        documents = [result['snippet'] for result in results]
        most_similar_document = gg.find_most_similar_documents(question, documents)
        document = most_similar_document
        paragraph = gg.the_best_paragraph(question, most_similar_document)
        context = document
        
        result = qa_pipeline({'question': question, 'context': context}) 
        st.session_state.question = question
        st.session_state.context = context
        st.text_area("Founded Context:", value=paragraph, height=150)
        st.text_area("Answer", value=result['answer'], height=80)


st.markdown('</div>', unsafe_allow_html=True)

    