import pandas as pd
import numpy as np

def get_keys(data):
    return [k for k, _ in data.items()]

def has_answer(df):
    return df[df.data.apply(get_keys).apply(lambda x: 'answer' in x)]

def get_context(data):
    return data['text']

def get_question(data):
    return data['question']

def get_answer_position(data):
    return data['short_candidate_start']

def get_answer(data):
    return data['short_candidate']

def question_context(df, i):
    question = df.data.apply(get_question)[i]
    text = test_df.context[i]
    print(
        f'- Question: {question}\n'
        f'- Context: {text}'
    )
    
def get_paragraph(df):
    return df.data.apply(lambda x: x['paragraphs'])
