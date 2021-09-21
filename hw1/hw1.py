import os
import nltk
import pandas as pd
import numpy as np
import random
import re
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")
mystem = Mystem()
vectorizer = CountVectorizer(analyzer='word')
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

# Функция индексирования данных.
# На выходе создает обратный индекс, он же матрица Term-Document.
def inverted_indexing(corpus):
    X = vectorizer.fit_transform(corpus)
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    return df


# a) какое слово является самым частотным
def most_popular_word(df) -> str:
    words = df.columns[df.sum(axis=0).to_numpy().argsort()[-10:]]
    ans_a = []
    for w in words:
        # число взято не просто так, я посмотрела, что максимальная частота встречаемости слова - 165
        if df[df[w] >0].shape[0] > 164:
            ans_a.append(w)
    return ans_a[random.randint(0, len(ans_a) - 1)]


# b) какое самым редким
def less_popular_word(df) -> str:
    ans_b = df.columns[df.sum(axis=0).to_numpy().argsort()[:100]]
    return ans_b[random.randint(0, len(ans_b) - 1)]


# c)какой набор слов есть во всех документах коллекции
def words_in_all_docs(df) -> str:
    ans_c = []
    for word in df.columns:
        k = df[word]
        if k[k>0].shape[0] == k.shape[0]:
            ans_c.append(word)
    return ', '.join(ans_c)


# d) кто из главных героев статистически самый популярный (упонимается чаще всего)?
def most_popular_character(df):
    max_name = ''
    max_int = 0
    for name in friends_names:
        if name.lower() in df.columns:
            if df[name.lower()].sum() > max_int:
                max_name = name
                max_int = df[name.lower()].sum()
    return max_name, max_int


if __name__ == "__main__":
    docs_texts, docs_paths = collect_docs(current_directory='../friends-data')
    corpus = preprocess(docs_texts)
    matrix = inverted_indexing(corpus)
    df = inverted_indexing(corpus)
    ans_a = most_popular_word(df)
    ans_b = less_popular_word(df)
    ans_c = words_in_all_docs(df)
    max_name, max_int = most_popular_character(df)
    print("Топ частотных слов несколько, я рандомайзером из их числа вывожу сейчас слово: ", ans_a)
    print("Топ самых нечастотных слов несколько, я рандомайзером из их числа вывожу сейчас слово: ", ans_b)
    print('Набор слов, которые есть во всех документах коллекции: ', ans_c)
    print('Самый популярный главный герой: ', max_name, ', он был упомянут {} раз'.format(str(max_int)))
