#2/7/2023, Kelsey Johnson
#Learn Python For Data Science #1 - Introduction by Siraj Raval
#https://www.youtube.com/watch?v=T5pRlIbr6gg
#Naive Bayes example on Github
#https://github.com/chribsen/simple-machine-learning-examples/blob/master/very_simple_examples/naive_bayes_classifier.py

#import Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

#set gaussian naive bayes classifier variable
gnb = GaussianNB()

#create list of data points [height (cm), weight (kg), shoe size (European)]
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47],
 [175,64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

#create classification variable for .fit() method
y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female',
 'male', 'female', 'male']

#train the model with .fit() method
gnb = gnb.fit(x,y)

#set prediction variable to store predicted result of entered data set
prediction = gnb.predict([[190,70,43]])



print(prediction)


