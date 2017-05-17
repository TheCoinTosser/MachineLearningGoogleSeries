###############################################
# Hello World - Machine Learning Recipes #1   #
# https://www.youtube.com/watch?v=cKxRvEZd3Mw #
###############################################

import sklearn.tree as tree
from common import common

from sklearn.preprocessing import LabelEncoder

# Read training set
trainingSet = common.read_csv(__file__, "fruits_training_set.csv")
lastIndex = len(trainingSet.columns) - 1

# Separate features and labels
features = trainingSet.iloc[:, 0:lastIndex]
labels = trainingSet.iloc[:, lastIndex]

labelEncoder = LabelEncoder()
labelsEncoded = labelEncoder.fit_transform(labels)

TEXTURE_INDEX = 1
feature_encoder = common.encode_label(features, TEXTURE_INDEX)

print("\n# Features")
for i in range(0, labels.size):
    print(features.iloc[i])

print("\n# Labels")
for i in range(0, labels.size):
    print("%s(%d)" % (labels.iloc[i], labelsEncoded[i]))


# Training
classifier = tree.DecisionTreeClassifier()
# "You can think of 'fit' as a synonym for 'find patterns in data'"
classifier.fit(features, labels)


# Predictions
predictions = common.read_csv(__file__, "fruits_predictions.csv")
common.encode_label(predictions, TEXTURE_INDEX, feature_encoder)

print("\n# Predictions")
print(classifier.predict(predictions))
