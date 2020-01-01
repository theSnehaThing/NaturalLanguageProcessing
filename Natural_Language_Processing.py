#Natural language Processing

#Importingg the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #removing punctuations,numbers
        review = review.lower() #setting all to lowercase
        review = review.split() #string to list
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # removing stopwords & took stem words only
        review = ' '.join(review) #reversing list back to string
        corpus.append(review)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)        
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler() 
X_train=sc_X.fit_transform(X_train) 
X_test=sc_X.transform(X_test) 

#fitting on train_set 
from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB() 
classifier.fit(X_train, y_train) 

#prediction on test_set 
y_pred = classifier.predict(X_test) 

#confusion matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 

