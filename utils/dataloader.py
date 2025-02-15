import pandas as pd
from tqdm import tqdm
import dataset as Dataset
import settings as preprocessing

configs = {
    
    'zalo' : '../data/zac2022_train_merged_final.json',
    'xsquad': '../data/dev_xsquad.json',
    'bert_url':'https://raw.githubusercontent.com/mailong25/bert-vietnamese-question-answering/master/dataset/train-v2.0.json',
    'custom': '../data/custom.json'
}



def merge_dataset():
    
    """ 
    Loading and Processing Zalo2022 Dataset
    """
    
    train_df = pd.read_json(configs['zalo'])
    questions = Dataset.has_answer(train_df).data.apply(Dataset.get_question).reset_index(drop=True)
    contexts = Dataset.has_answer(train_df).data.apply(Dataset.get_context).reset_index(drop=True)
    answers = Dataset.has_answer(train_df).data.apply(Dataset.get_answer).reset_index(drop=True)
    answer_starts = Dataset.has_answer(train_df).data.apply(Dataset.get_answer_position).reset_index(drop=True)
    
    cus_df = pd.read_csv('../data/custom.csv')

    
    zalo = pd.DataFrame(
        {
            'question': questions,
            'context': contexts,
            'answer': answers,
            'answer_start': answer_starts
        }
    )
    

    
    question = cus_df['question']
    context = cus_df['context']
    answer = cus_df['answer']
    answer_start = cus_df['answer_start']
    
    cus = pd.DataFrame(
        {
            'question': question,
            'context': context,
            'answer': answer,
            'answer_start': answer_start
        }
    )
    
    
    
    df_xsquad = pd.read_json(configs['xsquad'])
    df_xsquad.drop('version',axis = 1,inplace = True)    
    paragraphs = Dataset.get_paragraph(df_xsquad)
    dfs = []
    for p in paragraphs:
        p_df = pd.DataFrame(p)
        question_df = p_df['qas'].apply(lambda x: x[0]['question']).to_frame('question')
        start_pos_and_text_df = p_df['qas'].apply(lambda x: x[0]['answers'][0]).apply(pd.Series)
        final_p_df = pd.concat([question_df, p_df, start_pos_and_text_df], axis=1)
        final_p_df.drop('qas', axis=1, inplace=True)
        final_p_df.columns = ['question', 'context', 'answer_start', 'answer']
        dfs.append(final_p_df)
        
    xsquad = pd.concat(dfs, axis=0, ignore_index=True)
    
    df_bert = pd.read_json(configs['bert_url'])
    paras = Dataset.get_paragraph(df_xsquad)
    dfs = []
    for p in tqdm(paras):
        p_df = pd.DataFrame(p)
        p_df = p_df[p_df.qas.apply(lambda x: x[0]['answers'] != [])].reset_index(drop=True)
        if p_df.values.tolist() == []:
            continue
        question_df = p_df['qas'].apply(lambda x: x[0]['question']).to_frame('question')
        start_pos_and_text_df = p_df['qas'].apply(lambda x: x[0]['answers'][0]).apply(pd.Series)
        final_p_df = pd.concat([question_df, p_df, start_pos_and_text_df], axis=1)
        final_p_df.drop('qas', axis=1, inplace=True)
        final_p_df.columns = ['question', 'context', 'answer_start', 'answer']
        dfs.append(final_p_df)
        
    bert_qa = pd.concat(dfs, axis=0, ignore_index=True)
    
    qa_dataset = pd.concat([zalo,cus, xsquad, bert_qa], axis=0, ignore_index=True)
    qa_dataset.to_json('../data/qa_dataset.json')

def main():
    
    merge_dataset()
    train_df = pd.read_json('../data/qa_dataset.json')
    train_df = preprocessing.add_end_pos(train_df)
    train_df.to_json('../data/qa_dataset.json')
    
if __name__ == '__main__':
    main()
