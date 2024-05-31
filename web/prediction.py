import torch
from torch.nn.parameter import Parameter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoConfig, AutoModelForQuestionAnswering,XLMRobertaForQuestionAnswering, XLMRobertaConfig
import transformers


@torch.inference_mode()
def run_prediction(model, tokenizer, context, question):
    inputs = tokenizer.encode_plus(
        question, context, 
        return_tensors='pt'
    ).to('cpu')
    

    with torch.no_grad():
        outputs = model(**inputs)
    
    print(outputs)
    answer_start = torch.argmax(outputs[0])  
    answer_end = torch.argmax(outputs[1]) + 1 

    print(f'Answer: {tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))}')
    
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs.input_ids[0][answer_start:answer_end]
        )
    )
    
    print(
        f'- Question: {question}\n'
        f'- Answer: {answer}'
    )
    
    return question,answer
