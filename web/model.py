import torch
from torch.nn.parameter import Parameter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoConfig, AutoModelForQuestionAnswering,XLMRobertaForQuestionAnswering, XLMRobertaConfig, AutoTokenizer
import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_data
def load_model(model_path):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()
    return model
