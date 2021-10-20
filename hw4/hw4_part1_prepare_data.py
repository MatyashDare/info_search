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
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")
mystem = Mystem()
model_ft = gensim.models.KeyedVectors.load('../vectorizers/araneum_none_fasttextcbow_300_5_2018.model')
bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

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

def make_df_from_corpus(path_to_json:str):
    with open(path_to_json, 'r') as f:
        qa_corpus = list(f)[:11000]
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
    df['ans_lemmas'] = df['answers'].apply(preprocess_line)
    return df

def save_embed_corpus(df):
    df['ans_embeds'] = df['ans_lemmas'].apply(make_ft_embedding)
    ans_embeds = df['ans_lemmas'].apply(make_ft_embedding).to_numpy()
    split_df = pd.DataFrame(df['ans_embeds'].tolist(),
                        columns=[f'word{i}' for i in range(300)])
    split_df['doc_name'] = df['answers']
    split_df.to_csv('../corpora/corpus_ft.csv')

        
def save_bert_corpus(texts, model, tokenizer):
    vectors = []
    for text in texts:
      t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
      with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
      embeddings = model_output.last_hidden_state[:, 0, :]
      embeddings = torch.nn.functional.normalize(embeddings)
      vectors.append(embeddings[0].cpu().numpy())
    BERT_corpus = sparse.csr_matrix(vectors)
    sparse.save_npz('../corpora/BERT_answers.npz', BERT_corpus)
    return sparse.csr_matrix(vectors)


def get_query_bert(query):
    cls_embeddings = get_bert_corpus(query, b_model, b_tokenizer)
    return sparse.csr_matrix(cls_embeddings)



if __name__ == "__main__":
    df = make_df_from_corpus(path_to_json='../data/questions_about_love.jsonl')
    df = df.head(10000)
    save_embed_corpus(df)
    save_bert_corpus(df['answers'], bert_model, bert_tokenizer)
