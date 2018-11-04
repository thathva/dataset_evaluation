
"""
Dataset obtained from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
To classify tumors based on Malignant or Benign using Artifical Neural Networks with Keras
"""
#import libraries

import pandas as pd


#read dataset
df=pd.read_csv('data.csv')
#drop id and an unnamed column
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)

#split dataset into dependent and independent variables
X=df.iloc[:,1:].values
Y=df['diagnosis']

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#encode independent variable since it is a category ie M or B
label_encoder_y=LabelEncoder()
Y=label_encoder_y.fit_transform(Y)
onehotencoder=OneHotEncoder(categorical_features=[1])
Y=onehotencoder.fit_transform(Y).toarray()

#split the model into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

#preprocess the data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#build ann
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

#adding 2 hidden layers with output_dim as average of number of dependent and independent variables
#and using Rectifier function as activation function
classifier.add(Dense(output_dim=15,init='uniform',activation='relu',input_dim=30))

classifier.add(Dense(output_dim=15,init='uniform',activation='relu'))

#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#connect all neurons
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit the data
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=20)
#accuracy of 98%

#predict
y_pred=classifier.predict(X_test)

#since its probabilites, get False or True
y_pred=(y_pred>0.6)

#for accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred.round())

print(cm)
#cm= [[64,1],[2,47]]
(64+47)/114 #0.97

print(accuracy_score(y_test,y_pred.round()))
#accuracy of 97%
