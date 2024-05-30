# Question_Answering

## Instalation
Firstly, you need to create a new venv
```
python.exe -m pip install --upgrade pip
python -m venv myenv
myenv\Scripts\activate #Windows

#or Linux
source myenv/bin/activate
```

## Install needed libraries
```
pip install -r requirements.txt
python -m pip install -e .
```

## Prepare Data
Here, we combine 3 public datasets : Zalo2022,XSQuad and BertQA datasets.

```
cd utils
python dataloader.py
```


### Pretrained Model
You can download pretrained model hear [link](https://www.kaggle.com/code/tuanphong27/roberta-qa-fine-tuning/output)

## Usage
```
streamlit run app.py
```


<aside>
🎯 An application Question - Answering
</aside>

## Tasks:
- [x] Find data
- [x] Merge data
- [x] EDA
- [x] Report Introduction on Overleaf
- [ ] Design UI/UX
- [X] Fine-tuning model
- [ ] Choose metrics for model (F1,accuracy, pecission, recall)
- [X] Find model to help search context.      
- [ ] Deploy model


