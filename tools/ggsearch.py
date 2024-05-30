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
from sentence_transformers import SentenceTransformer, util

load_dotenv()

api_keys = [
    os.getenv("API_KEY1"),
    os.getenv("API_KEY2")
]

SearchEngineID = os.getenv("ENGINE_ID")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def find_most_similar_documents(question,documents):
    
    print(documents)
    document_embeddings = model.encode(documents)
    question_embedding = model.encode([question])[0]

    # Compute the cosine similarity between the question and each document
    similarities = util.pytorch_cos_sim(question_embedding, document_embeddings)

    # Find the most similar document
    most_similar_document_index = similarities.argmax()
    most_similar_document = documents[most_similar_document_index]
    return most_similar_document

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

        print(url,text)
        return text
    except:
        return ""
class GoogleSearch():
    
    def __init__(self):
        most_similar_documents = None
    
    def gg_search(self, query):
        pages_content = search("lang_vi " + query, num=5, stop=5, pause=2)
        document_urls = list(set(pages_content))

        with Pool(4) as p:
            ggsearch_results = p.map(getContent, document_urls)

        return document_urls, ggsearch_results

def main():
    question = "Quốc kỳ Việt Nam có mấy màu?"
    ggsearch = GoogleSearch()
    document_urls, ggsearch_results = ggsearch.gg_search(question)
    most_similar_document = find_most_similar_documents(question, ggsearch_results)
    print(most_similar_document)

if __name__ == "__main__":
    main()