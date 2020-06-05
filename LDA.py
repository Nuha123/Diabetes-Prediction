import numpy as np
import pandas as pd
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7,8]]
y = dataset.iloc[:, 9]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_test)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
print(X_train)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print('Accuracy  ' + str(accuracy_score(y_test, y_pred)))

