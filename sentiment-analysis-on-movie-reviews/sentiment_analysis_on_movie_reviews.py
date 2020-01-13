import numpy as np 
import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#import file into dataframe
df=pd.read_csv('TextFiles/moviereviews.tsv',sep='\t')
df.head()
#data cleaning
#
#remove NAs
df.dropna(inplace=True)
#remove empty string reviews
blanks=[] #stores index with blank reviews

for i,lb,rv in df.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)
blanks #w ehave blanks so we drop them
df.drop(blanks,inplace=True)
df['label'].value_counts()
sid=SentimentIntensityAnalyzer()
df['scores']=df['review'].apply(lambda review:sid.polarity_scores(review))
df['compound']=df['scores'].apply(lambda d:d['compound'])
df['comp_score']=df['compound'].apply(lambda score: 'pos' if score>=0 else 'neg')
df.head()
print(accuracy_score(df['label'],df['comp_score']))
print(confusion_matrix(df['label'],df['comp_score']))
print(classification_report(df['label'],df['comp_score']))
