import sklearn
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# reading dataset
my_property_data = pd.read_csv("NY Realstate Pricing.csv")

predict = "room_type"

# pre processing the data
le = preprocessing.LabelEncoder()
F1 = le.fit_transform(list(my_property_data["F1"])) # identifier
id = le.fit_transform(list(my_property_data["id"])) # identifier
neighbourhood = le.fit_transform(list(my_property_data["neighbourhood"])) # location
latitude = le.fit_transform(list(my_property_data["latitude"])) # location
longitude = le.fit_transform(list(my_property_data["longitude"])) # location
days_occupied_in_2019 = le.fit_transform(list(my_property_data["days_occupied_in_2019"]))
minimum_nights = le.fit_transform(list(my_property_data["minimum_nights"]))
number_of_reviews = le.fit_transform(list(my_property_data["number_of_reviews"]))
reviews_per_month = le.fit_transform(list(my_property_data["reviews_per_month"]))
availability_2020 = le.fit_transform(list(my_property_data["availability_2020"]))
price = le.fit_transform(list(my_property_data["price"])) # price in USD
room_type = le.fit_transform(list(my_property_data["room_type"]))
# 4 types 0 = house, 1 = hotel, 2= private, 3 = shared

# seperating attributes from target
x = list(zip(neighbourhood, latitude, longitude, days_occupied_in_2019, minimum_nights, number_of_reviews,
             reviews_per_month, availability_2020, price))
y = list(room_type)

num_folds = 5
seed = 7

scoring = "accuracy"

import sklearn.model_selection

# seperating dataset into test set and train set, 80-20 split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=seed)

# list of models used
predictive_models = []

# four models that will be tested for predictive accuracy
predictive_models.append(("NB", GaussianNB()))
predictive_models.append(("SVM", SVC()))
predictive_models.append(("RF", RandomForestClassifier()))
predictive_models.append(("GBM", GradientBoostingClassifier()))

model_results = []
model_names = []

# determining best model
print("Performance on Training set")
# go through model list and use each one
for name, model in predictive_models:
    kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
    # adding model name and accuracy to lists
    model_results.append(cv_results)
    model_names.append(name)
    # outputing results of model
    print(f"{name}: {cv_results.mean():,.6f} ({cv_results.std():,.6f})\n")

# bar graph comparison of models
fig = pyplot.figure()
fig.suptitle("Predictive Algorithm Comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(model_results)
ax.set_xticklabels(model_names)
pyplot.show()

predictive_models.append(("RF", RandomForestClassifier))
randf = RandomForestClassifier()

# assinging best model
best_model = randf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print(f"Best Model Accuracy Score on Test Set: {accuracy_score(y_test, y_pred)}")

print(classification_report(y_test, y_pred))

# creating confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
pyplot.show()

# outputting test results
for x in range(len(y_pred)):
    print(f"Predicted: {y_pred[x]} Actual: {y_test[x]} Data: {x_test[x]}")
