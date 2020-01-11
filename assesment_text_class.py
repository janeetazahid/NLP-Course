import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#import the file inot a data frame 
df=pd.read_csv('TextFiles/moviereviews2.tsv',sep='\t')
df.head()
#check for missing values
df.isnull().sum()
df.dropna(inplace=True)
#remove blank spaces 
blanks=[]
for i,lb,rv in df.itertuples(): #iterate over df for each row
    if rv.isspace(): #is the review is blank
        blanks.append(i) #add that row into blank list
len(blanks) #we see theres no blank spaces!
#look at label column 
df['label'].value_counts()
#split the data 
X=df['review']
y=df['label']
X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.33,random_state=42)
#create a pipline obj which vectorizes the data 
text_clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
#train model
text_clf.fit(X_train,y_train)
#make predictions
predictions=text_clf.predict(X_test)
#analyze models 
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))