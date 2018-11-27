import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Importing the dataset
st = "C:/Users/Monika/Desktop/Puns2/pUN.csv"
st.encode('utf-8')
dataset = pd.read_csv(st,encoding = "ISO-8859-1")

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []


for i in range(0, 3245):
    review = re.sub('[^a-zA-Z]', ' ', dataset['sentence'][i])
    #print (i)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model = Sequential()
model.add(Dense(32, input_dim=4000))
model.add(Activation('relu'))

