# Question_Answering

## Instalation
Firstly, you need to create a new venv
```
python -m venv myenv
myenv\Scripts\activate #Windows

#or Linux
source myenv/bin/activate
```

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

