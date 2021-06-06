import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                 names=["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                        "Flavanoids",
                        "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
                        "OD280/OD315 of diluted wines", "Proline"])


print(df)
print("-----------------------")
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(df.describe())
# print("-----------------------")


# 2 - Data Analysis

# Missing values check
 print(df.isnull().sum())
 print("-----------------------")


# Histograms
df.hist(bins=50, figsize=(20, 15))
plt.show()


# 3 - Data Processing

# Formatting the floats
float_formatter = "{:.10f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

# Dataset converted to numpy
dataset = df.to_numpy()

# Y values
Y = dataset[:, 0]
Y = Y.astype(int)

# X values
X = preprocessing.normalize(dataset[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])

# Create test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# 4 - Model Selection and Training

# Logistic Regression
lrclf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
lrclf.fit(x_train, y_train)

y_predicted = lrclf.predict(x_test)

print("Logistic Regression")
print("Actual   : ", end="")
print(y_test)
print("Predicted: ", end="")
print(y_predicted)
print()

accuracies = cross_val_score(estimator=lrclf, X=x_train, y=y_train, cv=10)
print("Accuracy: %", end="")
print(accuracies.mean() * 100)
print("Standard Deviation: %", end=""),
print(accuracies.std() * 100)
scores = cross_val_score(lrclf, x_train, y_train, scoring='neg_mean_absolute_error', cv=10)
print("MAE mean: %", scores.mean() * 100)
print()

# Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_predicted = clf.predict(x_test)

print("Decision Tree")
print("Actual   : ", end="")
print(y_test)
print("Predicted: ", end="")
print(y_predicted)
print()

accuracies = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10)
print("Accuracy: %", end="")
print(accuracies.mean() * 100)
print("Standard Deviation: %", end=""),
print(accuracies.std() * 100)
scores = cross_val_score(clf, x_train, y_train, scoring='neg_mean_absolute_error', cv=10)
print("MAE mean: %", scores.mean() * 100)
print()

# Naive Bayes
clf2 = GaussianNB()
clf2.fit(x_train, y_train)

y_predicted = clf2.predict(x_test)

print("Naive Bayes")
print("Actual   : ", end="")
print(y_test)
print("Predicted: ", end="")
print(y_predicted)
print()

accuracies = cross_val_score(estimator=clf2, X=x_train, y=y_train, cv=10)
print("Accuracy: %", end="")
print(accuracies.mean() * 100)
print("Standard Deviation: %", end=""),
print(accuracies.std() * 100)
scores = cross_val_score(clf2, x_train, y_train, scoring='neg_mean_absolute_error', cv=10)
print("MAE mean: %", scores.mean() * 100)
print()

# 5 - Fine-tune Model

param_grid = {'criterion': ['entropy', 'gini'], 'max_depth': np.arange(3, 10), 'min_samples_leaf': np.arange(1, 10)}

dtModel_grid = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=param_grid, verbose=1, cv=10, n_jobs=-1)

dtModel_grid.fit(x_train, y_train)

print(dtModel_grid.best_estimator_)
print(dtModel_grid.best_score_)

# 6 - Testing


y_pred = dtModel_grid.predict(x_test)

print()
print("Actual   :", y_test)
print("Predicted:", y_pred)
print()


print(confusion_matrix(y_test, y_pred,), ": is the confusion matrix")
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
print(precision_score(y_test, y_pred, average='macro'), ": is the precision score")
print(recall_score(y_test, y_pred, average='macro'), ": is the recall score")
print(f1_score(y_test, y_pred, average='macro'), ": is the f1 score")


