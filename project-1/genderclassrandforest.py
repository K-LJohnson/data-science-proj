#2/7/2023, Kelsey Johnson
#Learn Python For Data Science #1 - Introduction by Siraj Raval
#https://www.youtube.com/watch?v=T5pRlIbr6gg
#Scikit Randomized Forests
#https://scikit-learn.org/stable/modules/ensemble.html#forest


from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier()

#[height (cm), weight (kg), shoe size (European)]
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47],
 [175,64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]


y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female',
 'male', 'female', 'male']


rfc = rfc.fit(x,y)


prediction = rfc.predict([[170,57,38]])


print(prediction)

#Not as good as Gaussian Naive Bayes model. Has the same incorrect 
#prediction rate as the binary tree (50%).
