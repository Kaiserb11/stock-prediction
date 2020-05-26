import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("tesla.csv", sep=",", header=0)
data = data[['Open', 'High', 'Low', 'Volume', 'Close']]

predict = 'Close'

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# we write a try-except block to avoid repeated training
# if the pickle file is already available with the model dumped in it, we directly access it


try:
    pickle_in = open("stockmodel.pickle", "rb")
    clf = pickle.load(pickle_in)


except:
    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
    with open("stockmodel.pickle", "wb") as f:
        pickle.dump(clf, f)


prediction = clf.predict(x_test)


# here we calculate the accuracy by comparing the x_test values and the y_test values
# we also print the side by side comparision of the predicted and the actual values


acc = clf.score(x_test, y_test)
print("\naccuracy : ", acc)
print('\n\n Prediciton    Actual Value')
for x in range(len(prediction)):
    print(round(prediction[x], 6), '  ', round(y_test[x], 6))


# here you can try out values by giving a manual input to get a prediction on them
# put values in the list in the following order : 'Open', 'High', 'Low', 'Volume'


manual_data = np.array([[827, 843.29, 808, 15844000]])
manual_predict = clf.predict(manual_data)
print("\n\nclosing :  ", manual_predict)
