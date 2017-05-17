######################################################################################
# Train an Image Classifier with TensorFlow for Poets  - Machine Learning Recipes #6 #
# https://www.youtube.com/watch?v=cSKfRcEDGUs                                        #
#                                                                                    #
# This code relates to the code showed at 3:54. It has been tweaked to work with the #
# latest TensorFlow version instead.                                                 #
######################################################################################

from sklearn import metrics, model_selection
import tensorflow as tf
from tensorflow.contrib import learn

# Load dataset
iris = learn.datasets.load_dataset("iris")
x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data,
                                                                    iris.target,
                                                                    test_size=0.25,
                                                                    random_state=42)

# Specify that all features have real-value data
# dimension=n --> the training set has 'n' features
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
# print(feature_columns)

# 3 Layer DNN (Deep Neural Network) with 10, 20 and 10 neurons respectively.
#
# The 3 layers come from the length of the hidden_units array only. It has nothing to do with "n_classes=3".
#
# "n_classes = 3" means there are 3 types of outputs (the 3 types of flowers). I believe this parameter is used so it
# can set up the neural network in advance, before we train anything (otherwise the 3 classes could have been inferred
# from the total amount of classes in the "y_train" array, but that would require y_train to be looped twice, one to
# get the number of classes and another while training).

tensorflow_classifier = learn.DNNClassifier(hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            feature_columns=feature_columns,
                                            model_dir="tmp/iris_model")

# TensorFlow has separated the SciKit Learn interface from its own. Now, if you still want to use SciKit's interface
# (as this example does), you have to wrap tensorflow's classifier using SKCompat.
classifier = learn.SKCompat(tensorflow_classifier)

# Fit and Predict (using SciKit Learn interfaces)
classifier.fit(x_train, y_train, steps=200)
prediction = classifier.predict(x_test)
# print(prediction)

# Measure accuracy
score = metrics.accuracy_score(y_test, prediction["classes"])
print("Accuracy: {0:f}".format(score))
