'''
dataset downloaded from:  https://www.kaggle.com/rounakbanik/the-movies-dataset
'''
import pandas as pd

df=pd.read_csv('movies_metadata.csv')

df.head(5)

#simulating top 250 movies similar to that of IMBD's top 250
c=df['vote_average'].mean()

#using 85% quartile as cutoff for vote counts
co=df['vote_count'].quantile(0.85)

#now filtering based on co
movies=df.copy().loc[df['vote_count']>=co]

#to find the weighted average we use the formula v/(v+m) * R) + (m/(m+v) * C
def weighted(x,m=co,C=c):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
    
df['score']=df.apply(weighted,axis=1)

#sorting by score
df=df.sort_values(by='score',ascending=False)

#displaying top 250
df[['title','score']].head(250)
