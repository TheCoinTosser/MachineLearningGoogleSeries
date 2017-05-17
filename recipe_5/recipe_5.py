##############################################################
# Writing Our First Classifier - Machine Learning Recipes #5 #
# https://www.youtube.com/watch?v=AoeEHqVSNOw                #
#                                                            #
# Quotes and considerations:                                 #
#                                                            #
# Neural Network > Decision Tree > K-means  ( > = better )   #
#                                                            #
# Some features are more important than others, but K-means  #
# makes no distinction between them (ie, it treats them the  #
# same). As a consequence K-means works well when all of the #
# features have similar importance.                          #
##############################################################

from scipy.spatial import distance
import math


def dist(u, v):
    return distance.euclidean(u, v)


class Scrappy1MeansClassifier:

    def __1_means(self, X_test_single):

        prediction_single = 0
        min_sum = math.inf

        for i in range(len(self.X_train)):

            calculated_dist = dist(X_test_single, self.X_train[i])
            if calculated_dist < min_sum:
                min_sum = calculated_dist
                prediction_single = self.y_train[i]

        return prediction_single

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):

        prediction_array = []

        for X_test_single in X_test:

            predicted_label = self.__1_means(X_test_single)
            prediction_array.append(predicted_label)

        return prediction_array


from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)

classifier = Scrappy1MeansClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
# print(predictions)
print("Accuracy: %f%%" % (accuracy_score(y_test, predictions)*100))
