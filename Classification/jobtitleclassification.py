''' 
Classifying Job Titles using Python
Dataset downloaded from https://www.kaggle.com/san-francisco/sf-full-time-employees-by-job-classification/
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#read dataset
df=pd.read_csv("classification.csv")

#plot scatterplot between biweekly high and low rate
plt.scatter(df['Biweekly High Rate'],df['Biweekly Low Rate'])
plt.show()

#prints 97%. A very strong and only meaningful correlation in the dataset
print(df[['Biweekly High Rate','Biweekly Low Rate']].corr())

#drop FY column
df.drop(['FY'],axis=1,inplace=True)

#check for NaN values
df['Dept Level'].isnull().value_counts()

#store the features to be tested with
z=df.loc[:,df.columns!='Job Title']
x=np.array(z)

#store the column to be classified
y=df.iloc[:,2].values

#encode the values because dataset contains text
y=LabelEncoder().fit_transform(y)


x[:,0]=LabelEncoder().fit_transform(x[:,0])
x[:,1]=LabelEncoder().fit_transform(x[:,1])
x[:,2]=LabelEncoder().fit_transform(x[:,2])
x[:,3]=LabelEncoder().fit_transform(x[:,3])
x[:,4]=LabelEncoder().fit_transform(x[:,4])

#split test and train dataset with 30% as testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#fit training and testing dataset to RandomForest
clf=RandomForestClassifier(n_estimators=25,bootstrap=False)
clf.fit(x_train,y_train)

#predict
pred=clf.predict(x_test)

#additional- To view which Job title is mapped to which encoded labels
xy=LabelEncoder()
xy.fit(df['Job Title'])
mapping=dict(zip(xy.classes_,xy.transform(xy.classes_)))
#print(mapping)

#accuracy score
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,pred)
print(score)

#73% accuracy score. Unable to tune further due to memory limitations


