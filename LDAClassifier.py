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
lda = LDA()
lda.fit(X_train,y_train)
y_pred = lda.predict(X_test)
print(lda.score(X_test,y_test))





