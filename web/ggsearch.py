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
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')

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
        pages_content = search("lang_vi " + query, num=5, stop=5, pause=2)  # Decrease pause time
        document_urls = list(set(pages_content))

        with Pool(cpu_count()) as p:  # Use all available cores
            ggsearch_results = p.map(getContent, document_urls)

        return document_urls, ggsearch_results

def search_document(ggsearch,question):
    
    if(ggsearch == None):
        ggsearch = GoogleSearch()
    
    document_urls, ggsearch_results = ggsearch.gg_search(question)
    most_similar_document = find_most_similar_documents(question, ggsearch_results)
    most_similar_paragraph = the_best_paragraph(question, most_similar_document)
    return most_similar_document,most_similar_paragraph


def main():
    question = "Quốc kỳ Việt Nam có bao nhiêu màu?"
    ggsearch = None
    
    most_similar_document, most_similar_paragraph = search_document(ggsearch, question)

    print("Most similar document: ", most_similar_document)
    print("Most similar paragraph: ", most_similar_paragraph)
if __name__ == "__main__":
    main()