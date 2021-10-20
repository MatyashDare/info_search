import json
import gensim
from gensim.models.wrappers import FastText
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
import pickle
import scipy
from scipy import sparse
import torch
from transformers import AutoTokenizer, AutoModel
mystem = Mystem()
model_ft = gensim.models.KeyedVectors.load('../vectorizers/araneum_none_fasttextcbow_300_5_2018.model')
bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
df = pd.read_csv('../data/questions_about_love.csv')

def vec_normalization(vec):
    return vec / np.linalg.norm(vec)

def make_ft_embedding(line:str):
    emb_list = []
    for word in line.split():
        emb_list .append(model_ft[word])
    emb_list = vec_normalization(np.array(emb_list))
    return emb_list.mean(axis=0)


def preprocess_line(text: str) -> str:
    # убираем пунктуацию, оставляем только дефисы
    no_punct = re.sub('[,\?\!\."\:]|[a-zA-Z]+', '', ''.join(text))
    # токенизируем и лемматизируем текст, приводим к нижнему регистру
    lem_words = mystem.lemmatize(text.lower())
    ans = ' '.join([w for w in lem_words if w.isalpha()])
    if ans == " ":
        return 'пустой текст'
    elif ans == '':
        return 'пустой текст'
    else:
        return re.sub('\n', '', ans)


def cs_FastText(query, corpus_ft, cols):
    query = preprocess_line(query)
    scores = np.dot(corpus_ft[cols].values, make_ft_embedding(query).T)
    argx = np.argsort(scores)[::-1]
    return corpus_ft['doc_name'][argx.ravel()]


def cosine_similarity_matrix_query(sparse_matrix, query):
    return np.dot(sparse_matrix, query.T).toarray()


def make_bert_embedding(query):
    embeddings = embed_bert_cls(query, bert_model, bert_tokenizer)
    return sparse.csr_matrix(embeddings)


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].numpy()

def cs_BERT(query, sparse_matrix):
    query = make_bert_embedding(query)
    scores = cosine_similarity_matrix_query(sparse_matrix, query)
    argx = np.argsort(scores, axis=0)[::-1]
    return df['answers'][argx.ravel()]


def main(query:str, top_n:int):
    path_to_corpus = '../corpora/corpus_ft.csv'
    corpus_ft = pd.read_csv(path_to_corpus)
    cols = [col for col in corpus_ft.columns if 'word' in col]
    tf_answers = cs_FastText(query, corpus_ft, cols)[:top_n].to_numpy()
    sparse_matrix = sparse.load_npz('../corpora/BERT.npz')
    bert_answers = cs_BERT(query, sparse_matrix)[:top_n].to_numpy()
    return tf_answers, bert_answers

if __name__ == "__main__":
    query = input('Введите фразу, которую хотите найти в корпусе: ')
    top_n = int(input('Напишите число n, топ-n результатов выдачи моделей вы хотите видеть: '))
    tf_answers, bert_answers = main(query, top_n)
    tf_answers = ',\n'.join(tf_answers)
    bert_answers = ',\n'.join(bert_answers)
    print(f'Вот топ-{top_n} ответов, найденных с помощью FastText: \n', tf_answers, '\n\n')
    print(f'Вот топ-{top_n} ответов, найденных с помощью BERT: \n', bert_answers)
