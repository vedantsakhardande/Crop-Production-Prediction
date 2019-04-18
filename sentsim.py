from gensim import corpora, models, similarities
from collections import defaultdict
import pandas as pd
import numpy as np

def get_data():
    data = pd.read_csv("querydataset.csv",usecols = ['Query'])
    df=data.values
    X=df
    data = pd.read_csv("querydataset.csv",usecols = ['Answer'])
    df=data.values
    y=df
    # print("Y NEW SHAPE :",y.shape)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y = np.reshape(y, (y.shape[0]))
    return X, y

def give_answer():
    documents=[]
    X,y=get_data()
    for i in range(0,len(X)):
        check=X[i]
        check1=check[0]
        check2=check1[0]
        documents.append(check2)

    stoplist = set(['is', 'how'])

    texts = [[word.lower() for word in document.split()
            if word.lower() not in stoplist]
            for document in documents]

    # print(texts)
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1]
            for text in texts]
    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    doc = input("Enter the query :")
    vec_bow = dictionary.doc2bow(doc.lower().split())

    # convert the query to LSI space
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])

    # perform a similarity query against the corpus
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    print(sims)
    tup=sims[0]
    pos=tup[0]
    print("The Answer is :",y[pos])


if __name__ == "__main__":
    give_answer()
# (base) E:\Mini Project Work>"C:/Users/Vedant Sakhardande/Anaconda/python.exe" "e:/Mini Project Work/sentsim.py"
# paramiko missing, opening SSH/SCP/SFTP paths will be disabled.  `pip install paramiko` to suppress
# Enter the query :Which pesticide is used for rice cultivation in ratnagiri
# [(1, 1.0), (2, 0.93452114), (0, 0.7466595)]
# The Answer is : Monorin