import json
import gensim
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import re
import streamlit as st
from pymystem3 import Mystem
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import scipy
from scipy import sparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import time
import base64


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

def cosine_similarity_matrix_query(sparse_matrix, query):
    return np.dot(sparse_matrix, query.T).toarray()

def vec_normalization(vec):
    return vec / np.linalg.norm(vec)

def make_ft_embedding(line:str, model_ft):
    emb_list = []
    for word in line.split():
        emb_list .append(model_ft[word])
    emb_list = vec_normalization(np.array(emb_list))
    return emb_list.mean(axis=0)

def cs_FastText(query, corpus_ft, cols, model_ft, df):
    start_time = time.time()
    scores = cosine_similarity(make_ft_embedding(query, model_ft).reshape((1,300)), corpus_ft[cols].as_matrix())[0]
    argx = np.argsort(scores)[::-1]
    end_time = time.time()
    ans_time = end_time - start_time
    return df['answers'][argx.ravel()], ans_time

def cs_CountVectorizer(query, count_vectorizer, sparse_matrix, df):
    start_time = time.time()
    query = count_vectorizer.transform([query])
    scores = cosine_similarity_matrix_query(sparse_matrix, query)
    argx = np.argsort(scores, axis=0)[::-1]
    end_time = time.time()
    ans_time = end_time - start_time
    return df['answers'][argx.ravel()], ans_time


def cs_TfidfVectorizer(query, tfidf_vectorizer, sparse_matrix, df):
    start_time = time.time()
    query = tfidf_vectorizer.transform([query])
    scores = cosine_similarity_matrix_query(sparse_matrix, query)
    argx = np.argsort(scores, axis=0)[::-1]
    end_time = time.time()
    ans_time = end_time - start_time
    return df['answers'][argx.ravel()], ans_time


def cs_BM25(query, count_vectorizer, sparse_matrix, df):
    start_time = time.time()
    query = count_vectorizer.transform([query])
    scores = cosine_similarity_matrix_query(sparse_matrix, query)
    argx = np.argsort(scores, axis=0)[::-1]
    end_time = time.time()
    ans_time = end_time - start_time
    return df['answers'][argx.ravel()], ans_time

def make_bert_embedding(query, bert_model, bert_tokenizer):
    embeddings = embed_bert_cls(query, bert_model, bert_tokenizer)
    return sparse.csr_matrix(embeddings)


def cs_BERT(query, sparse_matrix, bert_model, bert_tokenizer, df):
    start_time = time.time()
    query = make_bert_embedding(query, bert_model, bert_tokenizer)
    scores = cosine_similarity_matrix_query(sparse_matrix, query)
    argx = np.argsort(scores, axis=0)[::-1]
    end_time = time.time()
    ans_time = end_time - start_time
    return df['answers'][argx.ravel()], ans_time

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].numpy()




@st.cache(allow_output_mutation=True)
def data_loader():
    d = {}
    bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    model_ft = gensim.models.KeyedVectors.load('../vectorizers/araneum_none_fasttextcbow_300_5_2018.model')
    df = pd.read_csv('../corpora/qa.csv')
    d['bert_tokenizer'] = bert_tokenizer
    d['bert_model'] = bert_model
    d['model_ft'] = model_ft
    d['df'] = df
    d['mystem'] = Mystem()
    
    count_vectorizer = pickle.load(open('../vectorizers/count_vectorizer.pickle', 'rb'))
    sparse_matrix_count_vectorizer = sparse.load_npz('../corpora/count_vectorizer_questions.npz')
    d['count_vectorizer'] = count_vectorizer
    d['sparse_matrix_count_vectorizer'] = sparse_matrix_count_vectorizer
    sparse_matrix_BM25 = sparse.load_npz('../corpora/BM25_questions.npz')
    d['sparse_matrix_BM25'] = sparse_matrix_BM25
    
    tfidf_vectorizer = pickle.load(open('../vectorizers/tfidf_vectorizer.pickle', 'rb'))
    sparse_matrix_tfidf_vectorizer = sparse.load_npz('../corpora/tfidf_vectorizer_questions.npz')
    d['tfidf_vectorizer'] = tfidf_vectorizer
    d['sparse_matrix_tfidf_vectorizer'] = sparse_matrix_tfidf_vectorizer
    
    corpus_ft = pd.read_csv('../corpora/corpus_ft_questions.csv')
    cols = [col for col in corpus_ft.columns if 'word' in col]
    d['corpus_ft'] = corpus_ft
    d['cols'] = cols
    
    sparse_matrix_BERT = sparse.load_npz('../corpora/BERT_questions.npz')
    d['sparse_matrix_BERT'] = sparse_matrix_BERT
    return d


    
def main():   
    st.title('🥰')
    st.title('LOVE SEARCH')
    st.header('Проблема с личной жизнью? Может, здесь получится найти ответ, близкий сердцу?')
    st.subheader('Корпус составлен на ответах Мейл.ру.  Он смешной, не берите ответы близко к сердцу, если они вам вдруг не понравятся')
    st.subheader('Мы ищем в корпусе вопросы, аналогичный вашему, и выдаем ответы на них')
    st.write('Ниже можно ввести ваш запрос и при желании послушать прекрасную музыку')
    audio_file = open('audio.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    query = st.text_area('Введите ваш запрос')
    choice = st.selectbox('Выберите метод, по которому будет происходить поиск',
                          ['CountVectorizer', 'TfidfVectorizer', 'BM25', 'FastText', 'BERT'])
    top_n = st.slider('Выберите число топ-н результатов поиска', 1, 50, 5)
    if st.button('искать'):
        if query == '':
            st.error('Не нужно вводить пустые запросы!')
        else:
            ans_time = -1000
            ans, ans_time = query_return(query, choice, top_n)
            ans_df = pd.DataFrame({'ответы': ans})
            st.write('Вот, что удалось найти (P.S.: можно таблицу увеличить в размере или же навести мышкой на понравившийся ответ и прочитать ешо полностью):')
            st.dataframe(ans_df)
            st.write(f"Все ответы нашлись за {ans_time} секунд!" )
            st.success('Поиск прошел без ошибок!')
            if ans_time != -1000:
                st.balloons()
    st.write('Если вам не нравятся ответы, можно нажать на кнопку "Видео" и посмотреть смешное видео')
    if st.button('Видео'):
        video_url  = 'https://www.youtube.com/watch?v=OGnDljE4R2g'
        st.video(video_url)

def query_return(query, choice, top_n):
    if choice == 'CountVectorizer':
        count_vectorizer = d['count_vectorizer']
        sparse_matrix = d['sparse_matrix_count_vectorizer']
        ans_ranged, time = cs_CountVectorizer(query, count_vectorizer, sparse_matrix, df)
    elif choice == "TfidfVectorizer":
        tfidf_vectorizer = d['tfidf_vectorizer']
        sparse_matrix = d['sparse_matrix_tfidf_vectorizer']
        ans_ranged, time = cs_TfidfVectorizer(query, tfidf_vectorizer, sparse_matrix, df)
    elif choice == "BM25":
        count_vectorizer = d['count_vectorizer']
        sparse_matrix = d['sparse_matrix_BM25']
        ans_ranged, time = cs_BM25(query, count_vectorizer, sparse_matrix, df)
    elif choice == "FastText":
        corpus_ft = d['corpus_ft']
        cols = d['cols']
        model_ft = d['model_ft']
        ans_ranged, time = cs_FastText(query, corpus_ft, cols, model_ft, df)
    elif choice == "BERT":
        sparse_matrix = d['sparse_matrix_BERT']
        bert_model = d['bert_model']
        bert_tokenizer = d['bert_tokenizer']
        ans_ranged, time = cs_BERT(query, sparse_matrix, bert_model, bert_tokenizer, df)
    return ans_ranged[:top_n].tolist(), time




def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('background.jpg')


d = data_loader()
df = d['df']
main()