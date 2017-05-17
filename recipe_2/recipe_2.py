##############################################################
# Visualizing a Decision Tree - Machine Learning Recipes #2  #
# https://www.youtube.com/watch?v=tNa99PG8hR8                #
##############################################################

from sklearn import datasets
from sklearn import tree
import numpy as np

iris = datasets.load_iris()

# Printing some info about the database
# print(iris.feature_names)
# print(iris.data[0])
# print(iris.target_names)
# print(iris.target[0])
#
# for i in range(len(iris.target)):
#     print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))


# The iris database is conveniently designed so the first 50 rows are test sets for one type of flower,
# the next 50 for the second type of flower and the next 50 for the third type. So, we can extract one sample of each
# type by getting the first index of each "set"
test_set_indexes = [0, 50, 100]

# Training data
training_data = np.delete(iris.data, test_set_indexes, axis=0)
training_target = np.delete(iris.target, test_set_indexes)

# Testing data
testing_data = iris.data[test_set_indexes]
testing_target = iris.target[test_set_indexes]

clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_target)

print(testing_target)
print(clf.predict(testing_data))


# viz code
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
