import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7,8]]
y = dataset.iloc[:, 9]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_test)

model = Sequential()
model.add(Dense(8, input_dim=8, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(5, input_dim=5, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(1, input_dim=1, kernel_initializer='random_uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y,epochs=100,batch_size=10)

scores = model.evaluate(X,y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))