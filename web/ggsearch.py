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
    most_similar_document = documents[0] + documents[1] + documents[most_similar_document_index]
    return most_similar_document

def the_best_paragraph(question, document):
    
    paragraphs = document.split('\n')
    best_paragraph = ""
    best_paragraph_score = 0
    for paragraph in paragraphs:
        if len(paragraph) < 10:
            continue
        sentences = sent_tokenize(paragraph)
        sentence_embeddings = model.encode(sentences)
        question_embedding = model.encode([question])[0]
        similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)
        max_similarity = similarities.max()
        if max_similarity > best_paragraph_score:
            best_paragraph_score = max_similarity
            best_paragraph = paragraph
    return best_paragraph
    

service = build("customsearch", "v1", developerKey=api_keys[0])

def get_google_results(query, num_results=10):
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

def main():
    question = "Thủ đô nước Pháp tên là gì?"
    documents = get_google_results(question)
    documents = [doc['snippet'] for doc in documents]
    most_similar_document = find_most_similar_documents(question,documents)
    best_paragraph = the_best_paragraph(question, most_similar_document)
    
if '__main__' == __name__:
    main()