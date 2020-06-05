# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7,8]]
y = dataset.iloc[:, 9]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_test)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0,gamma='auto')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_preds = classifier.predict(X_train)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print ("Recall - ",tp/(tp+fn))

#Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy using confusion matrix - ",accuracy_score(y_test, y_pred))

#Accuracy using AdaBoost
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
svc=SVC(probability=True, kernel='linear')
kfold = model_selection.KFold(n_splits=10, random_state=0)
model = AdaBoostClassifier(n_estimators=50, base_estimator=svc,random_state=0)
model1 = model.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print("Accuracy using adaboost:",accuracy_score(y_test, y_pred))
