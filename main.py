import sys
sys.path.append('../')
import joblib
import requests
import streamlit as st
import tensorflow as tf
from types import SimpleNamespace
import wikipedia 
import openai
import gradio as gr
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from web.model import load_model
import importlib
from tools import ggsearch as gg
from tools.ggsearch import GoogleSearch,find_most_similar_documents

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
model = load_model('./model/qa_large_model/')

question = "Quốc kỳ Việt Nam có mấy màu?"
context = ""

ggsearch = GoogleSearch()
document_urls, ggsearch_results = ggsearch.gg_search(question)
most_similar_document = find_most_similar_documents(question, ggsearch_results)
context += most_similar_document

print(f"Question: {question}")
print(f"Context: {context}")

question, answer = predict(model, tokenizer, context, question)
