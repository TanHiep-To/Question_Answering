import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.nn.parameter import Parameter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoConfig, AutoModelForQuestionAnswering


def add_end_pos(df):
    len_answers = df.answer.apply(len)
    df['answer_end'] = df.answer_start + len_answers
    return df

def add_token_positions(encodings, start_pos, end_pos):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, start_pos[i]))
        end_positions.append(encodings.char_to_token(i, end_pos[i]-1)) 
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length 
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length 
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
