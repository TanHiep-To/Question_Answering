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
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
from googlesearch import search
import time
from wrapt_timeout_decorator import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

api_keys = [
    os.getenv("API_KEY1"),
    os.getenv("API_KEY2")
]

SearchEngineID = os.getenv("ENGINE_ID")

def find_most_similar_documents(question,documents):
    vectorizer = TfidfVectorizer().fit(documents)
    document_vectors = vectorizer.transform(documents)
    question_vector = vectorizer.transform([question])
    
    max_similarity = 0
    most_similar_documents = None

    for i in range(len(documents)):
        similarity = cosine_similarity(document_vectors[i],question_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_documents = (documents[i], question)

    return most_similar_documents

def getContent(url):
    try:
        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'lxml')


        for invisible_elem in soup.find_all(['script', 'style']):
            invisible_elem.extract()

        paragraphs = [p.get_text() for p in soup.find_all('p')]


        text = ' '.join(paragraph.strip() for paragraph in paragraphs)

        text = text.replace('\xa0', ' ')
        text = re.sub(r'\[\d+\]', '', text)
        text = text.replace('\n', '') 

        return text
    except:
        return ""
class GoogleSearch():
    
    def __init__(self):
        most_similar_documents = None
    
    def gg_search(self, query):
        pages_content = search("lang_vi " + query, num=10, stop=10, pause=2)
        document_urls = list(set(pages_content))

        with Pool(4) as p:
            ggsearch_results = p.map(getContent, document_urls)

        return document_urls, ggsearch_results
