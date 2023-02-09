#2/7/2023, Kelsey Johnson
#Learn Python For Data Science #1 - Introduction by Siraj Raval
#https://www.youtube.com/watch?v=T5pRlIbr6gg

#import decision tree
from sklearn import tree

#sample variable [height (cm), weight (kg), shoe size (European)] 
#(what we are training the model on)
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47],
 [175,64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

#classification variable (what the answer you are looking for is)
y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female',
 'male', 'female', 'male']

#variable for decision tree classifer (clf) model
clf = tree.DecisionTreeClassifier()

#Train the model with fit method (fitting the model to the data) 
#(takes 2 variables). After the model is trained, we can use a .predict() method
#call to predict if data points will be male or female.
clf = clf.fit(x,y)

#Our prediction variable to store the prediction from .prediction() method
prediction = clf.predict([[190,70,43]])

#I would be [170,57,38], my result should be female
#Print the prediction
print(prediction)