import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")
mystem = Mystem()


# функция с реализацией подсчета близости запроса и документов корпуса,
# на выходе которой вектор,
# i-й элемент которого обозначает близость запроса с i-м документом корпуса
def similarity_array(query, df):
    return cosine_similarity(query, df[cols].values)

def preprocess_doc(text: str) -> str:
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

#главная функция, объединяющая все это вместе;
# на входе - запрос,
# на выходе - отсортированные по убыванию имена документов коллекции
def main(query_input:str):
    df = pd.read_csv('doc_term_matrix.csv')
    cols = [col for col in df.columns if col != 'docs_paths' and col != 'doc_name']
    vect = pickle.load(open('friends_data_tfidf.pickle', 'rb'))
    query = vect.transform([preprocess_doc(query_input)])
    sim_array = cosine_similarity(query, df[cols].values)
    ans_list = df.loc[sim_array.argsort()[0][::-1]]['doc_name'].to_list()
    return ans_list

if __name__ == "__main__":
    query_input = input('Введите фразу, которое хотите найти в корпусе: ')
    print(main(query_input))
