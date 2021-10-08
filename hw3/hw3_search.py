import json
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")
mystem = Mystem()
from scipy import sparse
import pickle
count_vectorizer = pickle.load(open('./cv.pickle', 'rb'))
corpus_matrix = pickle.load(open('./sparse_matrix.pickle', 'rb'))


def preprocess_line(text: str) -> str:
    # убираем пунктуацию, оставляем только дефисы
    no_punct = re.sub('[,\?\!\."\:]|[a-zA-Z]+', '', ''.join(text))
    # токенизируем и лемматизируем текст, приводим к нижнему регистру
    lem_words = mystem.lemmatize(text.lower())
    ans = ' '.join([w for w in lem_words if w not in russian_stopwords and w.isalpha()])
    return re.sub('\n', '', ans)

def count_query(query:str, corpus_matrix, doc_name):
    query = preprocess_line(query)
    query_vec = count_vectorizer.transform([query])
    scores = corpus_matrix.dot(query_vec.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    ans = doc_name[sorted_scores_indx.ravel()]
    return ans

def main(query:str):
    df = pd.read_csv('./df.csv')
    doc_name = pd.read_csv('./df.csv')['doc_name'].to_numpy()
    ans = 'Найденные в порядке релевантности документы: ' + ', '.join(count_query(query, corpus_matrix, doc_name))
    return ans


if __name__ == "__main__":
    query = input('Введите фразу, которую хотите найти в корпусе: ')
    print(main(query))
