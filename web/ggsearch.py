from googleapiclient.discovery import build
from numpy import random
import requests
from bs4 import BeautifulSoup
from timeout_decorator import timeout
from nltk import sent_tokenize
from multiprocessing import Pool,Process,TimeoutError
import re
import timeout_decorator
import sys
from multiprocessing import cpu_count
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
from googlesearch import search
import time
from wrapt_timeout_decorator import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import nltk
import asyncio
import time
import aiohttp
from aiohttp.client import ClientSession
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
load_dotenv()

api_keys = [
    os.getenv("API_KEY1"),
    os.getenv("API_KEY2")
]


SearchEngineID = os.getenv("ENGINE_ID")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def find_most_similar_documents(question,documents):
    
    # Encode the question, so that we can use it to search for similar documents
    document_embeddings = model.encode(documents)
    question_embedding = model.encode([question])[0]

    # Compute the cosine similarity between the question and each document
    similarities = util.pytorch_cos_sim(question_embedding, document_embeddings)

    # Find the most similar document
    most_similar_document_index = similarities.argmax()
    most_similar_document = documents[most_similar_document_index]
    return most_similar_document

def the_best_paragraph(question, document):
    paragraphs = sent_tokenize(document)
    paragraph_embeddings = model.encode(paragraphs)
    question_embedding = model.encode([question])[0]

    similarities = util.pytorch_cos_sim(question_embedding, paragraph_embeddings)

    most_similar_paragraph_index = similarities.argmax()
    most_similar_paragraph = paragraphs[most_similar_paragraph_index]
    return most_similar_paragraph

service = build("customsearch", "v1", developerKey=api_keys[0])

def get_google_results(query, num_results=20):
    results = []
    for i in range(0, num_results, 10):
        res = service.cse().list(q=query, cx=SearchEngineID, start=i+1).execute()
        results += res['items']
    return results

def get_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except:
        return ""


# question = st.text_input("Enter your question:")
# if st.button("Search"):
#     results = get_google_results(question)
#     documents = [result['snippet'] for result in results]
#     most_similar_document = find_most_similar_documents(question, documents)
#     st.write(most_similar_document)
#     st.write("The best paragraph:")
#     best_paragraph = the_best_paragraph(question, most_similar_document)
#     st.write(best_paragraph)

# # Path: web/app.py