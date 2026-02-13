# Fake-News-Detection
 This project builds a Fake News Detection system using NLP and Machine Learning. News articles are preprocessed using stopword removal and stemming, then converted into numerical features using TF-IDF. A Logistic Regression model is trained to classify articles as Real (0) or Fake (1), achieving accurate predictions on test data.
 About the Dataset:

      1.id: unique id for a news article
      2.title: thetitle of a news article
      3.author:author of the news article
      4.text:the text of the article; could be incomplete
      5.label: a label that marks whether the news article is real or fake:

          1: Fake news

          0: real news

Importing the dependencies

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# printing the stopwordsin english
print(stopwords.words('english'))

Data pre-processing

news_dataset = pd.read_csv('/content/train.csv')

news_dataset.shape

# printing the first five data
news_dataset.head()

# counting the number of missing values
news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['title']+''+news_dataset['author']

print(news_dataset['content'])

# separating the data label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

print(X)
print(Y)

Stemming:

the process of reducing a word to its root word

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]','', content) #all the num and punctuation marks are removed from the content and replaced by '' space
  stemmed_content = stemmed_content.lower() # all the letters are convertd to lower case
  stemmed_content = stemmed_content.split() #converting into list
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # converting all words into root words and by using for loop removing the stopwords
  stemmed_content = ' '.join(stemmed_content) # join all the words in stemmed content
  return stemmed_content

# apply stemming function to our content colm
news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

# separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values


print(X)
print(Y)

Y.shape

# converting the textual data to numerical data
vectorizer = TfidfVectorizer() # tf checks the repeation of word and makes it an important word or idf makes that word not significant
vectorizer.fit(X) # fitting the vectorizer function to X

X = vectorizer.transform(X) # transform function coverts all these values to their respective festures

print(X)

Splitting the dataset into training and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, stratify=Y, random_state=2 )


Training the model: Logistic Regression

model = LogisticRegression()

model.fit(X_train, Y_train)

Evaluation

accuracy score

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy )

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

Making a Predictive System


X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if(prediction[0]==0):
  print('The news is real')
else:
  print('The news is fake')


print(Y_test[3])

🤝 Contributing

Contributions are welcome!
Feel free to fork this repository and submit a pull request.

📜 License

This project is open-source and available under the MIT License.
