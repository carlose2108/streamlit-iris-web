import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

# Load data
iris = load_iris()

X = iris.data
y = iris.target

# Training and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

svc_model = SVC()
svc_model.fit(X_train, y_train)

with open('linear_regression.pkl', 'wb') as li:
    pickle.dump(lr_model, li)

with open('logistic_regression.pkl', 'wb') as lg:
    pickle.dump(lg_model, lg)

with open('svc.pkl', 'wb') as svc:
    pickle.dump(svc_model, svc)