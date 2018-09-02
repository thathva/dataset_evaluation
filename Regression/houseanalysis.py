''' 
Predicting house rates using Regression on Python 3.6
Dataset downloaded from https://www.kaggle.com/harlfoxem/housesalesprediction
'''
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#read the dataset
df=pd.read_csv('kc_house_data.csv')

#display first 5 rows
df.head(5)

#clean the data by checking for NaN for each columns
df['price'].isnull().value_counts()
#repeat for other columns

#from the dataset, there seems to be a relation between the grade and the price
df[['price','grade']].corr()

#the correlation is fairly strong with 66%
#there is a stronger correlation between the price and the sqft_living of 70%
df[['price','sqft_living']].corr()

#on further analysis, we come to the conclusion that grade,sqft_living,bedrooms,bathrooms,floors,views,sqft_living15,sqft_above
#affects the price fairly strongly
#so these 8 variables can be our independent variables to predict the dependent variable price

#after deleting the columns having a weak correlation and not affecting the prices,
#store the prices as dependent variable in the variable 'prices'
prices=df.iloc[:,0].values

#store independent variables
z=df.loc[:,df.columns!='price']
X=np.array(z)

#Encode independent variables
X[:,1]=LabelEncoder().fit_transform(X[:,1])
X[:,2]=LabelEncoder().fit_transform(X[:,2])
X[:,3]=LabelEncoder().fit_transform(X[:,3])
X[:,4]=LabelEncoder().fit_transform(X[:,4])
X[:,5]=LabelEncoder().fit_transform(X[:,5])
X[:,6]=LabelEncoder().fit_transform(X[:,6])
X[:,7]=LabelEncoder().fit_transform(X[:,7])

#training and testing data split
x_train,x_test,y_train,y_test=train_test_split(X,df['price'],test_size=0.2)

#fit the classifer
clf=LinearRegression.fit(x_train,y_train)

#predict with test set
pred=clf.predict(x_test)

#r2 score
from sklean.metrics import r2_score
r2_score(y_test,pred)

#outputs 53%

#check the ols matrix using statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm

x2=sm.add_constant(X)
est=sm.OLS(df['price'],x2)
est2=est.fit()
est2.summary()



