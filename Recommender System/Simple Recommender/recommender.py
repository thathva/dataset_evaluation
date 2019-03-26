'''
dataset downloaded from
https://www.kaggle.com/rounakbanik/the-movies-dataset
'''

import pandas as pd
df=pd.read_csv('C:/Users/PGDM//Desktop/movies_metadata.csv')

#content based recommendation
#we use the overview column to extract words so that movies can be recommended
from sklearn.feature_extraction.text import TfidfVectorizer
df=df[:10000]

tfidf=TfidfVectorizer(stop_words='english')
df['overview']=df['overview'].fillna('')
tfidf_mat=tfidf.fit_transform(df['overview'])

#since tfidf is used, cosine can be used for similarity score
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_mat, tfidf_mat)

#to identify index based on title
idx=pd.Series(df.index,index=df['title']).drop_duplicates()

#function to return recommendations
def recommendations(title,cosine_sim=cosine_sim):
    ind=idx[title]
    scores=list(enumerate(cosine_sim[ind]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores=scores[1:16]
    movieind=[i[0] for i in scores]
    return df['title'].iloc[movieind]

recommendations('The Lord of the Rings: The Fellowship of the Ring')

''' 
Output
2007                            The Lord of the Rings
8785                                       The Hobbit
7000    The Lord of the Rings: The Return of the King
5814            The Lord of the Rings: The Two Towers
9560                                    Underclassman
2709                                     On the Ropes
1185                                      Raging Bull
4213              What's the Worst That Could Happen?
3877                   Planes, Trains and Automobiles
6436                                  White Lightning
6347                                          Extasis
4031                                The Price of Milk
8761                                         Mistress
4147                                            Krull
1941                              Herbie Goes Bananas


'''
