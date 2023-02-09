#2/7/2023, Kelsey Johnson
#Learn Python For Data Science #1 - Introduction by Siraj Raval
#https://www.youtube.com/watch?v=T5pRlIbr6gg
#Scikit Nearest Neighbors Unsupervised model
#https://scikit-learn.org/stable/modules/neighbors.html

#import KNeighbors classifier
from sklearn.neighbors import KNeighborsClassifier

#initialize the classifier variable and set size of neighborhood as 3 
nnu = KNeighborsClassifier(n_neighbors=3)

#create list of data points [height (cm), weight (kg), shoe size (European)]
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47],
 [175,64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

#create classification variable for .fit() method
y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female',
 'male', 'female', 'male']

#train the model with .fit() method
nnu = nnu.fit(x,y)

#intialize prediction variable to store predicted result
prediction = nnu.predict([[190,70,43]])

#print the result
print(prediction)

#Incorrect 100% of the time (with test only 2 data points). Worst predictor.