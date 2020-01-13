This assigement used text classification to determine is a movie review is positive or negative.

In order to determine this the data from the file **moviereviews2.tsv** was split into a traning and testing set.

Then a pipline object was created, which consisted of a **Tfidf vectorization** object and a **linear SVC** object.

The purpose of the **TfidfVectorizer** was:
  - count the number of times a term occures in the document 
  - multiply this by logarithm of the number of times the world occures in all documents 
 
The purpose of the **Linear SVC (support vector class)** was:
  - fit to the data and provides a best fit, which categorizes the data 

The pipeline object was used to train the model and thne used to predict the test values.

After this the model was evaluated using the follows metrics:
  - **confusion matrix**
  - **accuracy score**
  - **classification report** 
 
**Language(s)**: Python

**Software**: VSCode

**Libraries**: pandas, numpy, sklearn
