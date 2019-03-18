''' SMS based spam classifer.
Dataset downloaded from https://www.kaggle.com/team-ai/spam-text-message-classification
'''


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#read dataset
df=pd.read_csv('textmsgs.csv')

df.head()

#check for null values
df.info()

'''def redefine(column):
    for i in df[column]:
        if(i=='ham'):
            return 0
        else:
            return 1
'''

#changing to 0 and 1
df['Category']=df['Category'].map({'ham':0,'spam':1})

#visualizing in wordcloud for hamwords
hamwords=' '.join(list(df[df['Category']==0]['Message']))
wc=WordCloud(width=512,height=512).generate(hamwords)
plt.figure(figsize=(10,10))
plt.imshow(wc)
plt.show()

#for spamwords
spamwords=' '.join(list(df[df['Category']==1]['Message']))
wc2=WordCloud(width=512,height=512).generate(spamwords)
plt.figure(figsize=(10,10))
plt.imshow(wc2)
plt.show()
#most common word is free

#import countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#split data
X_train,X_test,y_train,y_test=train_test_split(df['Message'],df['Category'],test_size=0.2)
count=CountVectorizer()

#fit into the vectorizer
train=count.fit_transform(X_train)
test=count.transform(X_test)

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(train,y_train)

#fit data
pred=clf.predict(test)

#check accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))

#prints 0.9901
