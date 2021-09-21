import os
import nltk
import pandas as pd
import numpy as np
import random
import re
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")
mystem = Mystem()
vectorizer = TfidfVectorizer(analyzer='word', norm='l2')
friends_names = ['Фиби', 'Фибс', 'Моника', 'Мон', 'Рэйчел', 'Рэйч', 'Чендлер', 'Чен', 'Чэндлер', 'Росс', 'Джоуи', 'Джои', 'Джо']

def collect_docs(current_directory:str):
    i = 0
    docs_texts = {}
    docs_paths = {}
    for root, dirs, files in sorted(os.walk(current_directory)):
        for name in files:
            if name.endswith('.txt'):
                fpath = os.path.join(root, name)
                with open(fpath, 'r') as f:
                    text = f.read().splitlines()
                docs_texts[i] = text
                docs_paths[i] = fpath
                i += 1
    return docs_texts, docs_paths


def preprocess_doc(text: list) -> str:
    text = ' '.join(text)
    # убираем пунктуцию, оставляем только дефисы
    no_punct = re.sub('[,\?\!\."\:]|[a-zA-Z]+', '', ''.join(text))
    # убираем отдельно разметку диалога
    no_speech = re.sub(' - ', ' ', no_punct)
    # убираем технические детали
    no_tech = re.sub('(\ufeff|9999 00000500 --> 0000200 wwwtvsubtitlesnet)', '',no_speech)
    # убираем оставшиеся неприятности с пробелами и тд
    text = re.sub("\s\s+", " ", no_tech)
    # токенизируем и лемматизируем текст, приводим к нижнему регистру
    lem_words = mystem.lemmatize(text.lower())
    ans = ' '.join([w for w in lem_words if w not in russian_stopwords and w.isalpha()])
    return re.sub('\n', '', ans)


# Функция препроцессинга данных. Включите туда лемматизацию,
# приведение к одному регистру, удаление пунктуации и стоп-слов.
def preprocess(docs_texts):
    corpus = []
    for doc_name, doc in docs_texts.items():
        corpus.append(preprocess_doc(doc))
    return corpus


# функция индексации корпуса, 
# на выходе которой посчитанная матрица Document-Term
def doc_term_matrix(corpus):
    data = vectorizer.fit_transform(corpus).toarray()
    vocab = vectorizer.get_feature_names()
    df_ans = pd.DataFrame(data=data,columns=vocab)
    return df_ans


#считаем, что в current directory лежат .py файлы и датасет
def prepare_db_vec(cur_dir:str):
    docs_texts, docs_paths = collect_docs(current_directory=cur_dir)
    corpus = preprocess(docs_texts)
    df = doc_term_matrix(corpus)
    #  получаем Document-Term матрицу
    df['docs_paths'] = [docs_paths[i] for i in range(len(docs_paths))]
    df['doc_name'] = df['docs_paths'].apply(lambda x: re.findall('\d - (.*?).ru.txt', x)[0])
    # сохраняем Document-Term матрицу с двумя еще колонками - пути до файлов и их названиями
    # p.s.: пути до файлов не очень-то нужно хранить, но я подумала, что может пригодиться
    df.to_csv('doc_term_matrix.csv', index=False)
    # сохраняем векторайзер для дальнейшего использования
    pickle.dump(vectorizer, open('friends_data_tfidf.pickle', 'wb'))

if __name__ == "__main__":
    prepare_db_vec('./friends-data')
