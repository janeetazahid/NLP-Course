The purpose of this project was to perform sentiment analysis on several movie reviews and determine if it the review is 
positive or negative. 

In order to do this a textfile with movie reviews imported as a pandas dataframe.

Then the data was cleaned, any NA values were removed and any blank reviews were omitted.

After this the **SentimentIntensityAnalyzer** from the **NLTK** library was used to compute the **compound score** of each review.
Based on this if the compound score was positive it was categorized as a 'pos' review and if the score was negative it was categorized 
as a 'neg' review.

Finally, the computed label was compared to the pre-exsisting lables and the following metrics were used to evaluate the outcome:
- **accuracy score**

- **confusion matrix**

- **classification report**


**Language(s)**: Python 

**Software**: VSCode

**Libraries**: Numpy, Pandas, NLTK, sklearn
