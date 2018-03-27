# -*- coding: utf-8 -*-
"""
Created on Fri Feb  18 15:33:27 2018

@author: Dan Popoutanu
Reg number: 1704805

"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import time

start = time.time()
#get data without headers
train_data = pd.read_csv('C:/Users/Popoutanu/Desktop/CE807/Assignment1/train.tsv',header = 0, delimiter='\t', quoting=3)
test_data = pd.read_csv('C:/Users/Popoutanu/Desktop/CE807/Assignment1/test.tsv',header=0,delimiter='\t',quoting=3)

#train_data.info()

#words = [w for w in words if not w in stopwords.words("english")]
num_reviews = train_data["Phrase"].size

#creating the pipeline with bigram , tfidf and linearSVC

clf = Pipeline([('vect',CountVectorizer(analyzer="word",ngram_range=(1, 2),tokenizer=word_tokenize)),
                ('tfidf',TfidfTransformer()),
                ('clf',LinearSVC())])

#define each dataset for training and testing 
y=train_data['Sentiment']
x=train_data['Phrase']
#testing dataset
z=test_data['Phrase']

#starting the training function
clf.fit(x,y)
#predicting on test
data = clf.predict(z)
#computer the arithmetic mean over data
np.mean(data == z)
#print the score for refference 
print(clf.score(x,y))
#creating the csv
output = pd.DataFrame( data={"PhraseId":test_data["PhraseId"], "Sentiment":data} )
output.to_csv( "ce807_assignemtn1_dan_popoutanu.csv", index=False, quoting=3 )

#tracking how much time it took
end = time.time()
print('this took only : ' ,end-start, ' seconds')

