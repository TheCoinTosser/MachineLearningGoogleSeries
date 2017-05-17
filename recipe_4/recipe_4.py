#########################################################
# Let’s Write a Pipeline - Machine Learning Recipes #4  #
# https://www.youtube.com/watch?v=84gqSbLcBFE           #
#########################################################

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
classifier = tree.DecisionTreeClassifier() #KNeighborsClassifier()

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
# print(predictions)
print(accuracy_score(y_test, predictions))
