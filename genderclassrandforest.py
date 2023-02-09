#2/7/2023, Kelsey Johnson
#Learn Python For Data Science #1 - Introduction by Siraj Raval
#https://www.youtube.com/watch?v=T5pRlIbr6gg
#Scikit Randomized Forests
#https://scikit-learn.org/stable/modules/ensemble.html#forest

#import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

#initialize Random Forest classifier variable
rfc = RandomForestClassifier()

#create list of data points [height (cm), weight (kg), shoe size (European)]
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47],
 [175,64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

#create classification variable for .fit() method
y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female',
 'male', 'female', 'male']

#train the model with the .fit() method
rfc = rfc.fit(x,y)

#initialize prediction variable to store model prediction result (which is based
# on the entered data set)
prediction = rfc.predict([[170,57,38]])

#print the predicted result
print(prediction)

#Not as good as Gaussian Naive Bayes model. Has the same incorrect 
#prediction rate as the binary tree (50%), but I only tested 2 data points.