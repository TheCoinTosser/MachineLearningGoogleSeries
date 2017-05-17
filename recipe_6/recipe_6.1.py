#################################################################################################################
# Train an Image Classifier with TensorFlow for Poets - Machine Learning Recipes #6                             #
# https://www.youtube.com/watch?v=cSKfRcEDGUs                                                                   #
#                                                                                                               #
# Quotes                                                                                                        #
# "TensorFlow is especially useful for working with a branch of machine learning called deep learning."         #
# "In Deep Learning, you don't need to extract features manually. That is why it is so useful in image          #
# classification tasks since it is very hard to extract image features by hand"                                 #
# "The classifier used in deep learning is called Neural Network"                                               #
# "(...) a Neural Network is just another type of classifier, like the nearest neighbor one we wrote last time" #
# "Inception: One of Google's best image classifiers. It is Open Source."                                       #
# "Retraining (aka Transfer Learning): train new training data on top of an existing trained set."              #
#################################################################################################################

import tensorflow as tf
import os

image_test_set_folder = 'flower_test_set'

for file_name in os.listdir(image_test_set_folder):

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_test_set_folder + '/' + file_name, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("trained_images_outputs/retrained_labels.txt")]

    # Read the new .pb file which represents the new trained model
    with tf.gfile.FastGFile("trained_images_outputs/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print("\nFile name: %s" % file_name)
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            if human_string in 'metermaid':
                if score * 100 > 75:
                    print('%s winner %s' % (score * 100, human_string))
                    message = '%s chance %s was seen' % (score * 100, human_string)
                    print(message)
