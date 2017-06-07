# Machine Learning - Google Series (created by Josh Gordon)

This repo contains all of the source code needed in order to run the examples of all Machine Learning recipes from the Google series you can find here: https://www.youtube.com/watch?v=cKxRvEZd3Mw

I've taken the liberty to make a few modifications along the way, such as:

* Using Python3 instead of Python2.
* Fixing warnings and upgrading deprecated functions. I've used the latest versions of everything (Python3, SciKit Learn, TensorFlow, etc).
* Adding some quotes as well as considerations so we don't miss the important points.
* Adding extra code (for example, to visualize something in the data, to import data from files rather than hard-code it, etc).
* Getting rid of Docker. Some of the recipes use a pre-built Docker image in order to make things easier. However, I believe one learns better when they set up something by themselves. I've replaced Docker with new instructions instead, whenever Docker was needed in the original series.


**1. Recipe 1 -  Hello World**

* Goal: Create a classifier to predict between apples and oranges.
* Category: Binary classification.
* Changes made: The original video has the datasets hardcoded. I've taken the training and the test datasets off the code and put it into .csv files instead. Moreover, instead of using raw numbers directly to encode the features/labels, I've used SciKit's LabelEncoder() class instead. That way the data is presented in a nicer form and you don't need to know which feature/label is mapped to which number.


**2. Recipe 2 -  Visualizing a Decision Tree**

* Goal: Train an iris dataset using a Decision Tree classifier and visualize its inner model.
* Changes made: No substantial changes were made.


**3. Recipe 3 -  What makes a good feature?**

* Goal: Help you visualize whether or not a feature is useful, and if so, when it is useful. Also, the video gives you an insight on why multiple features are almost always necessary in order to have a good prediction.
* Category: Binary classification.
* Changes made: No substantial changes were made.


**4. Recipe 4 -  Let’s Write a Pipeline**

* Goal: Teach you how to partition your original dataset into training and test sets as well as measuring how good your predictions are using accuracy. It also gives you an insight on how linear binary classification works.
* Category: Multiclass classification.
* Changes made: No substantial changes were made.


**5. Recipe 5 - Writing Our First Classifier**
* Goal: Write your own classifier (1 nearest neighbour classifier)
* Category: Multiclass classification.
* Changes made: I've written my own version of 1 nearest neighbour so the implementation might be slightly different than the one shown in the video.


**6. Recipe 6 - Train an Image Classifier with TensorFlow for Poets**
* Goal: Train our first classifier that will take as input raw images and predict between 5 types of flowers.
* Category: Multiclass classification.
* Changes made: I have not used a Docker container in order to set-up everything needed. Instead, I've written a readme.txt file with instruction on how to download, install and configure everything from scratch. Finally, for the testing dataset, I've added 4 tests for each type of flower. For each type, the first image contains the flower by itself. The second contains a bunch of flowers. The third is a cartoonish/drawing and the forth is an image I figure would be hard for the classifier to get it right.


**7. Recipe 7 - Classifying Handwritten Digits with TF.Learn**
* Goal: Train a classifier to predict digits from 0 to 9 using TensorFlow.
* Category: Multiclass classification.
* Changes made: Again, I have not used the docker container to get tensorflow. Instead I've decided to compile TensorFlow myself to get better performance. If you don't want to do that, there are easier ways to get TensorFlow (https://www.tensorflow.org/install/). Finally, I've just made a few minor changes to the code in order to make it work with the latest version of TensorFlow.
