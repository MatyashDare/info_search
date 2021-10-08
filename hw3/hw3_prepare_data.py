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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import pickle

def preprocess_line(text: str) -> str:
    # убираем пунктуацию, оставляем только дефисы
    no_punct = re.sub('[,\?\!\."\:]|[a-zA-Z]+', '', ''.join(text))
    # токенизируем и лемматизируем текст, приводим к нижнему регистру
    lem_words = mystem.lemmatize(text.lower())
    ans = ' '.join([w for w in lem_words if w not in russian_stopwords and w.isalpha()])
    return re.sub('\n', '', ans)


def make_df(filename:str):
    curr_dir = os.getcwd()
    filename = os.path.join(curr_dir, filename)
    with open(filename, 'r') as f:
        qa_corpus = list(f)[:50000]
    questions = []
    answers = []
    for qa in qa_corpus:
        qa = json.loads(qa)
        if qa['answers'] != []:
            max_value = -10 ** 6
            max_text = ''
            for answer in qa['answers']:
                if answer['author_rating']['value'] != '':
                    cur_value = int(answer['author_rating']['value'])
                    if cur_value >= max_value:
                        max_text = answer['text']
            if max_text != '':
                answers.append(max_text)
                questions.append(qa['question'])
    df = pd.DataFrame({'questions': questions, 'answers': answers})
    df['lemmas'] = df['answers'].apply(preprocess_line)
    df['doc_name'] = [f'doc_{i}' for i in range(df.shape[0])]
    df.to_csv('./df.csv', index=False, encoding='utf-8')
    return df

def doc_term_matrix(corpus, tfidf_vectorizer, tf_vectorizer):
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    corpus_vocab = tfidf_vectorizer.get_feature_names()
    tf_matrix = tf_vectorizer.fit_transform(corpus)
    cv_matrix = count_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    idf = sparse.csr_matrix(idf)
    tf = tf_vectorizer
    pickle.dump(idf, open('./idf.pickle', 'wb'))
    pickle.dump(tf, open('./tf.pickle', 'wb'))
    pickle.dump(count_vectorizer, open('./cv.pickle', 'wb'))
    return tf_matrix, idf, cv_matrix

def BM_25(tf_matrix, idf_matrix, cv_matrix, k=2, b=0.75):
    values = []
    rows = []
    cols = []
    len_d = cv_matrix.sum(axis=1).T
    avgdl = len_d.mean()
    for i, j in zip(*tf_matrix.nonzero()):
        A = idf_matrix[0,j] * tf_matrix[i, j] * (k+1)
        B_1 = (k * (1 - b + b * len_d[0,i] / avgdl))
        B_1 = np.expand_dims(B_1, axis=-1) 
        B = tf_matrix[i, j] + B_1
        B = B[0]
        values.append(A/B)
        rows.append(i)
        cols.append(j)
    sparse_matrix = sparse.csr_matrix((values, (rows, cols)))
    pickle.dump(sparse_matrix, open('./sparse_matrix.pickle', 'wb'))
    return sparse_matrix


if __name__ == "__main__":
    df = make_df('questions_about_love.jsonl')
    corpus = df.lemmas.to_list()
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, analyzer='word', norm='l2')
    tf_matrix, idf_matrix, cv_matrix = doc_term_matrix(corpus, tfidf_vectorizer, tf_vectorizer)
    BM_25(tf_matrix, idf_matrix, cv_matrix)
