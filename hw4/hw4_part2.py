import pickle
import scipy
from scipy import sparse
import numpy as np
import pandas as pd
import ast



def get_cosine_similarity(sparse_matrix, query):
    return np.dot(sparse_matrix, query.T).toarray()
def get_cosine_similarity2(sparse_matrix, query):
    return np.dot(sparse_matrix, query.T)


def count_metric(method):
    if method != 'corpus_ft':
        anwers_matrix = sparse.load_npz(f'../corpora/{method}_answers.npz')
        questions_matrix = sparse.load_npz(f'../corpora/{method}_questions.npz')
        res_mat = get_cosine_similarity(anwers_matrix, questions_matrix)
    else:
        answers_matrix = pd.read_csv('../corpora/corpus_ft_answers.csv')
        questions_matrix = pd.read_csv('../corpora/corpus_ft_questions.csv')
        cols = [col for col in answers_matrix.columns if 'word' in col]
        res_mat = get_cosine_similarity2(answers_matrix[cols].values, questions_matrix[cols].values)
    return np.argsort(-res_mat, axis=1)

def count_score(result, top_n):
    score = 0
    for i, row in enumerate(result):
            top_results = row[:top_n]
            if i in top_results:
                score += 1
    return score/len(result)



def main():
    top_n = int(input('Введите параметр, метрику по топ-скольким запросам вы хотите посчитать: '))
    methods = ['count_vectorizer', 'tfidf_vectorizer', 'BM25', 'corpus_ft', 'BERT']
    dict_topn = {}
    for method in methods:
        result = count_metric(method)
        score = count_score(result, top_n)
        dict_topn[method] = score
    print(f'Выведем скоры по топ-{top_n} первых запросов: ')
    for key in dict_topn:
        if key == 'corpus_ft':
            key2 = 'FastText'
            print(f'{key2} --- {dict_topn[key]}')
        else:
            print(f'{key} --- {dict_topn[key]}')
    print(f'Выведем зараннее посчитанные скоры по топ-5 первых запросов: ')
    file = open("./dict_top5.txt", "r")
    contents = file.read()
    dict_top5_saved = ast.literal_eval(contents)
    for key in dict_top5_saved:
        if key == 'corpus_ft':
            key2 = 'FastText'
            print(f'{key2} --- {dict_top5_saved[key]}')
        else:
            print(f'{key} --- {dict_top5_saved[key]}')
            
if __name__ == "__main__":
    main()
    
# top-5    
# count_vectorizer --- 0.0336
# tfidf_vectorizer --- 0.0877
# BM25 --- 0.0685
# BERT --- 0.0083
# FastText --- 0.0022
