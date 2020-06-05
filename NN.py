# Importing the libraries
import pandas as pd
import numpy as np

from keras.models import Sequential

from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7,8]].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#initializing the ANN

classifier = Sequential()

#Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim= 5, init='uniform', activation='relu', input_dim= 8))

#Adding the second hidden layer

classifier.add(Dense(output_dim= 5, init='uniform', activation='relu'))

#Adding the output layer
model = Sequential()
classifier.add(Dense(output_dim= 1, init='uniform', activation='sigmoid'))

#compiling the ANN

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting the ANN to the training set

classifier.fit(X_train,y_train,epochs=100,batch_size=10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix for Test set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy for Test set
from sklearn.metrics import accuracy_score
test_accu=accuracy_score(y_test, y_pred)
print("Accuracy using ANN - ",accuracy_score(y_test,y_pred))


