import pandas as pd 
import re 
import numpy as np 
import nltk 
from nltk.corpus import stopwords
import kagglehub
from pathlib import Path

from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

if __name__=="__main__":
    nltk.download('punkt') # download model that recognizes typical sentences
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    
    df = pd.read_csv("history.csv")
    df.head()

    sentences = []
    for line in df['text']:
        sentences.append(sent_tokenize(line))

    sentences = [a for b in sentences for a in b]

    path = kagglehub.dataset_download("danielwillgeorge/glove6b100dtxt")

    home_dir = Path.home()
    word_embeddings = {}
    f = open(str(home_dir)+"\.cache\kagglehub\datasets\danielwillgeorge\glove6b100dtxt\\versions\\1\glove.6B.100d.txt",
             encoding='utf-8')
    for line in f:
        words = line.split()
        word = words[0]
        coffs = np.asarray(words[1:], dtype='float32')
        word_embeddings[word] = coffs
    f.close()

    cleaned_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ") # remove non-Latin letters
    cleaned_sentences = [s.lower() for s in cleaned_sentences]
    stop_words = stopwords.words('english')

    cleaned_sentences = [remove_stopwords(a.split()) for a in cleaned_sentences]

    sentence_vectors = []
    for i in cleaned_sentences:
      if len(i) != 0:
        j = sum([word_embeddings.get(a, np.zeros((100,))) for a in i.split()])/(len(i.split())+0.001)
      else:
        j = np.zeros((100,))
      sentence_vectors.append(j)

    sim_matrix = np.zeros([len(sentences), len(sentences)])

    for a in range(len(sentences)):
      for b in range(len(sentences)):
        if a != b:
          sim_matrix[a][b] = cosine_similarity(sentence_vectors[a].reshape(1,100), sentence_vectors[b].reshape(1,100))[0,0]


    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    for i in range(5):
      print("Original text:")
      print(df['text'][i])
      print('\n')
      print("Summary:")
      print(ranked_sentences[i][1])
      print('\n')
