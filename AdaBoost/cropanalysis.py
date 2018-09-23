'''
Analyzing and Predicting Total Production of coffee using AdaBoostRegressor
Dataset downloaded from: https://www.kaggle.com/sbajew/icos-crop-data
'''

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read csv
df=pd.read_csv('C:/Users/harip/Desktop/ICO_CROP_DATA.csv')

#check for NaN
df['TOTAL_PRODUCTION'].isnull().value_counts()
df['DOMESTIC_CONSUMPTION'].isnull().value_counts()
df['EXPORTABLE_PRODUCTION'].isnull().value_counts()
df['GROSS_OPENING_STOCKS'].isnull().value_counts()

#all have NaNs. Fill it with mean of the values

df['EXPORTABLE_PRODUCTION'].fillna(df['EXPORTABLE_PRODUCTION'].mean(),inplace=True)
df['DOMESTIC_CONSUMPTION'].fillna(df['DOMESTIC_CONSUMPTION'].mean(),inplace=True)
df['TOTAL_PRODUCTION'].fillna(df['TOTAL_PRODUCTION'].mean(),inplace=True)
df['GROSS_OPENING_STOCKS'].fillna(df['GROSS_OPENING_STOCKS'].mean(),inplace=True)

#now, checking for correlation if the data is related in any way
df[['TOTAL_PRODUCTION','DOMESTIC_CONSUMPTION']].corr()
#90% a very strong correlation

df[['TOTAL_PRODUCTION','EXPORTABLE_PRODUCTION']].corr()
#97%

df[['TOTAL_PRODUCTION','GROSS_OPENING_STOCKS']].corr()
#73%

df[['EXPORTABLE_PRODUCTION','GROSS_OPENING_STOCKS']].corr()
#67%

df[['EXPORTABLE_PRODUCTION','DOMESTIC_CONSUMPTION']].corr()
#80%

df[['GROSS_OPENING_STOCKS','DOMESTIC_CONSUMPTION']].corr()
#77%
#this concludes that all the factors are important to analyze and predict

#to analyze highest production by country
highestprod=df.groupby('COUNTRY')['TOTAL_PRODUCTION'].agg(sum)

#plot a bar graph
plt.figure(figsize=(20,20))
highestprod.plot(kind='bar')
plt.xlabel("COUNTRY")
plt.ylabel("TOTAL PRODUCTION")

#print out the highest production country and value
highestcountry=highestprod.idxmax()
highesttotal=highestprod.loc[highestprod.idxmax()]

print("Country:", highestcountry) 
print("Total:", highesttotal)
#Country: Brazil
#Total: 1118452.3505000002

#print out the lowest production country and value
lowestcountry=highestprod.idxmin()
lowesttotal=highestprod.loc[highestprod.idxmin()]

print("Country:", lowestcountry) 
print("Total:", lowesttotal)
#Country: Benin
#Total: 1.855

#similary domestic production by country
highestdomestic=df.groupby('COUNTRY')['DOMESTIC_CONSUMPTION'].agg(sum)

#plot out the bar graph
plt.figure(figsize=(20,20))
highestdomestic.plot(kind='bar')
plt.xlabel("COUNTRY")
plt.ylabel("DOMESTIC CONSUMPTION")

#print out highest domestic consumption country
domesticcountry=highestdomestic.idxmax()
domestictotal=highestdomestic.loc[highestdomestic.idxmax()]

print("Country:", domesticcountry) 
print("Total:", domestictotal)
#Country: Brazil
#Total: 419545.0

#group by country to see visualize how much total production is gone into domestic and exports
consump_prod=df.groupby('COUNTRY')[['TOTAL_PRODUCTION','DOMESTIC_CONSUMPTION','EXPORTABLE_PRODUCTION']].agg(sum)

#plot out the bar graph
plt.figure(figsize=(10,10))
consump_prod.plot(kind='bar')
plt.xlabel("COUNTRY")
plt.ylabel("Total Consumption")
#from the graph it is clear that the production is very high than the domestic consumption and domestic consumption is
#more than what can be exported.

#now, predicting the total production
#plot a scatter plot to see the dependent value
plt.scatter(df['DOMESTIC_CONSUMPTION'],df['TOTAL_PRODUCTION'])
plt.scatter(df['EXPORTABLE_PRODUCTION'],df['TOTAL_PRODUCTION'])

#a positive linear relation between the variables

#import train_test_split
from sklearn.model_selection import train_test_split

#store dependent variable in Y
Y=df['TOTAL_PRODUCTION']

#store independent variables in X, selected selectively because less number of columns
X=df[['DOMESTIC_CONSUMPTION','EXPORTABLE_PRODUCTION','GROSS_OPENING_STOCKS']]

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

#import AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

#fit the model
clf=AdaBoostRegressor()
clf.fit(x_train,y_train)

#predict the values
pred=clf.predict(x_test)

#see r2 score and mean absolute error
from sklearn.metrics import r2_score,mean_absolute_error

print(r2_score(y_test,pred))
#outputs 0.97 ie 97%

print(mean_absolute_error(y_test,pred))
#outputs 832




