The purpose of this assigment was to cluster qoura questions, based on the words in the questions.

The first step was to import the csv file which containted the questions. 

Then a **TfidfVectorizer** instance was used to transform the questions column.

After this **Non-Negative Maxtrix Decomposition** was used to fit the transformed data

From this the top 15 words for all 20 components were showcased.

Finally each question in the dataset was assigned a topic.

**Languages:** Python

**Software:** Jupyter Notebook 

**Libraries:** Pandas, sklearn


