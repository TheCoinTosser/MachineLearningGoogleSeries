##############################################################################
# Classifying Handwritten Digits with TF.Learn - Machine Learning Recipes #7 #
# https://www.youtube.com/watch?v=Gj0iyo265bc                                #
#                                                                            #
# Quotes and considerations:                                                 #
#                                                                            #
# "The images in MNIST are properly segmented, which means each image        #
# contains exactly one digit."                                               #
#                                                                            #
# "A 28 x 28 image has 784 features"                                         #
#                                                                            #
# "The images in the MNIST database come in flat, that means they are        #
# represented by a vector instead of a matrix. That is why if one needs to   #
# display an image, it needs to use the "reshape" method, like we did in the #
# display_test_image() function bellow."                                     #
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import learn


def display_test_image(i):
    test_img = images_test[i]
    plt.title('Image: %d. Label: %d' % (i, labels_test[i]))
    plt.imshow(test_img.reshape(28, 28), cmap='gray_r')
    plt.show()


tf.logging.set_verbosity(tf.logging.WARN)

print("Loading mnist database")

# Training data (55k images)
mnist = learn.datasets.load_dataset("mnist")
images_training = mnist.train.images
labels_training = np.asarray(mnist.train.labels,
                             dtype=np.int32)

# Test data (10k images)
images_test = mnist.test.images
labels_test = np.asarray(mnist.test.labels,
                         dtype=np.int32)

# You can print some images to see how they are using display_test_image
# display_test_image(0)

# Building and train our classifier
feature_columns = learn.infer_real_valued_columns_from_input(images_training)
tensorflow_classifier = learn.LinearClassifier(n_classes=10,
                                               feature_columns=feature_columns)

classifier = learn.SKCompat(tensorflow_classifier)

classifier.fit(x=images_test,
               y=labels_test,
               batch_size=100,
               steps=1000)

# Evaluate accuracy
prediction = classifier.score(images_test, labels_test)
print("Accuracy: %f" % prediction['accuracy'])
