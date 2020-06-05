# Importing the libraries
import pandas as pd
import sklearn


# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7,8]]
y = dataset.iloc[:, 9]
feature_cols = ['pregnancies', 'PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(X_test)
print(y_pred)

sklearn.tree.export_graphviz(classifier,out_file='file.dot')


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print ("Recall - ",tp/(tp+fn))

#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
print("Accuracy using confusion matrix - ",accuracy_score(y_test,y_pred))
#
# # Bagged Decision Trees for Classification
# from sklearn import model_selection
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
# kfold = model_selection.KFold(n_splits=10, random_state=0)
# cart = DecisionTreeClassifier()
# model = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=0)
# results = model_selection.cross_val_score(model, X, y, cv=kfold)
# print("Accuracy Using Bagging - ",results.mean())
#
# #Accuracy using AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# model = AdaBoostClassifier(n_estimators=100, random_state=0)
# results = model_selection.cross_val_score(model, X, y, cv=kfold)
# print("Accuracy using AdaBoost - ",results.mean())

