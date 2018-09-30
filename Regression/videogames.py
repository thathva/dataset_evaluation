'''
Predicting Global Sales of Video Games based on different regions using Random Forest Regressor 
https://www.kaggle.com/gregorut/videogamesales
'''

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the file
df=pd.read_csv('vgsales.csv')

#check for any missing values
df['Year'].isnull().value_counts()
df['Publisher'].isnull().value_counts()

#let's see the most games published by which Publisher and the Year where most games were released
df['Publisher'].value_counts()
df['Year'].value_counts()
#from the number of value counts, Electronic Arts as the publisher and 2009 as the year, we can fill the NaN with these values
df['Publisher'].fillna('Electronic Arts',inplace=True)
df['Year'].fillna(2009.0,inplace=True)

#find highest publihser in North America
publisher=df.groupby('Publisher')['NA_Sales'].agg('sum').sort_values(ascending=False)

#plot data
publisher[:50].plot(kind='bar')
plt.figure(figsize=(30,30))

#who is the publisher with highest sales in North America?
highest=publisher.idxmax()
highestcount=publisher.loc[publisher.idxmax()]

print(highest)
print(highestcount)
#Nintendo , 816

#In Japan?
publisherjp=df.groupby('Publisher')['NA_Sales'].agg('sum').sort_values(ascending=False)

#plot data
publisherjp[:50].plot(kind='bar')
plt.figure(figsize=(30,30))

highestjp=publisherjp.idxmax()
highestcountjp=publisherjp.loc[publisherjp.idxmax()]
#Nintendo, 415

#In Europe?
publishereu=df.groupby('Publisher')['EU_Sales'].agg('sum').sort_values(ascending=False)

#plot data
publishereu[:50].plot(kind='bar')
plt.figure(figsize=(30,30))

highesteu=publishereu.idxmax()
highestcounteu=publishereu.loc[publishereu.idxmax()]

print(highesteu,highestcounteu)
#Nintendo, 418

#Globally?
publishergl=df.groupby('Publisher')['Global_Sales'].agg('sum').sort_values(ascending=False)

#plot data
publishergl[:50].plot(kind='bar')
plt.figure(figsize=(30,30))

highestgl=publishergl.idxmax()
highestcountgl=publishergl.loc[publishergl.idxmax()]

print(highest,highestcount)
#Nintendo, 1786

#Most sold game
name=df.groupby('Name')['Global_Sales'].agg('sum').sort_values(ascending=False)

#plot data
name[:50].plot(kind='bar')
plt.figure(figsize=(30,30))

highestg=name.idxmax()
highestcountg=name.loc[name.idxmax()]

print(highestg,highestcountg)
#Wii Sports, 82.74

#by platform and Genre?
platform=df.groupby('Platform')['Global_Sales'].agg('sum').sort_values(ascending=False)
genre=df.groupby('Genre')['Global_Sales'].agg('sum').sort_values(ascending=False)

#plot data
platform.plot(kind='bar')
plt.figure(figsize=(30,30))

#plot data
genre.plot(kind='bar')
plt.figure(figsize=(30,30))

print(platform.idxmax())
print(platform.loc[platform.idxmax()])
print(genre.idxmax())
print(genre.loc[genre.idxmax()])
#PS2, 1255
#Action, 1751

#Now to predict global sales
df[['NA_Sales','Global_Sales']].corr()
#94%

df[['EU_Sales','Global_Sales']].corr()
#90%

df[['EU_Sales','Global_Sales']].corr()
#61%

df[['Other_Sales','Global_Sales']].corr()
#74%

#regression plot between all the sales
sns.regplot(df['NA_Sales'],df['Global_Sales'])
plt.xlabel("NA Sales")
plt.ylabel("Global Sales")

sns.regplot(df['EU_Sales'],df['Global_Sales'])
plt.xlabel("EU Sales")
plt.ylabel("Global Sales")

sns.regplot(df['JP_Sales'],df['Global_Sales'])
plt.xlabel("JP Sales")
plt.ylabel("Global Sales")

sns.regplot(df['Other_Sales'],df['Global_Sales'])
plt.xlabel("Other Sales")
plt.ylabel("Global Sales")

#store dependent and independent variables
Y=df['Global_Sales']
X=df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']]

#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#split the data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

#fit the model
reg=RandomForestRegressor()
reg.fit(x_train,y_train)

#predict the model
pred=reg.predict(x_test)

#plot against testing data
sns.regplot(pred,y_test)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
#displays almost perfect fit line

#import metrics
from sklearn.metrics import r2_score
from scipy.stats import pearsonr,spearmanr

print(r2_score(y_test,pred))
#97% accuracy
print(pearsonr(y_test,pred))
#99%, 0.0

print(spearmanr(y_test,pred))
#corr=99% pvalue=0.0






