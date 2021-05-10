import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df= pd.read_csv('/kaggle/input/weight-height/weight-height.csv')
df.head() ##prints a few lines of data
df.isnull().sum() #no null values... good to go... lucky
#df.Height.plot(kind="hist",title='Univariate: Height Histogram', color='c');#just a graph call

## with mat plot lib now...
#plt.hist(x=df.Height,color='c')
#plt.title("Univariate:Height Histogram")
#plt.xlabel("Height")
#plt.ylabel("Total Counts")
#plt.plot();

#df.Weight.plot(kind="hist",title='Univariate:Weight Histogram',color='c'); #another graph call

#weight histogram using matplotlib
#plt.hist(x=df.Weight, color='c')
#plt.title("Univariate:Weight Histogram")
#plt.xlabel("Weight")
#plt.ylabel("Total Counts")
#plt.plot();

#scatter plot
#plt.scatter(x=df["Height"], y=df["Weight"], color='c')
#plt.title("Bivariate: Height Vs Weight Using Matplotplib")
#plt.xlabel("Height")
#plt.ylabel("Weight")
#plt.plot();

#X=df.iloc[:,1:2].values
#y=df.iloc[:,2:3].values
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=31)
#from sklearn.linear_model import LinearRegression
#regressor= LinearRegression()
#model_fit=regressor.fit(X_train, y_train)
#y_predict=regressor.predict(X_test)
#y_predict

#scatter plot of updated data
#plt.scatter(X_test, y_test)
#plt.plot( X_train, regressor.predict(X_train), color='r')
#plt.xlabel("Height")
#plt.ylabel("Weight")
#plt.title("Heigth Vs Weight Prediction")
#plt.show()

#red points are predicted... blue points are actual values

## fro accuracy there are many different functions we can call...
#from sklearn.metrics import r2_score
#print(f"model ACcuracy is: {regressor.score(X_test,y_test)}")
#means we have 85 percent accuracy

#END OF CNN MODEL... THIS NEXT MODEL IS A KNN... IT PREDICTS GENDER BASED ON HEIGHT AND WEIGHT

#X_ml = df.iloc[:, 1:3].values
#y_ml = df.iloc[:, 0:1].values

#X_ml.shape
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#y_ml = encoder.fit_transform(y_ml)
#y_ml
#y_ml.shape
#X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.3, random_state=31)
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=3) #n_neighbors is the number of neighbours
#knn.fit(X_train_ml, y_train_ml)
#y_predict_ml = knn.predict(X_test_ml)
#y_predict_ml
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test_ml, y_predict_ml)
#print("accuracy:", knn.score(X_test_ml, y_test_ml))
#accuracy is 90 percent for the knn... this is a imporvement...
#from sklearn.model_selection import GridSearchCV
# below is the params dictionary object. Here we can add the parameters which we can experiment with. so now our model will get evaluated with all the possible
# combination of below params and let you the best score and param combination.
#params = {
 #   "n_neighbors": [5, 10, 20],
  #  'leaf_size' : [30, 40, 50],
   # 'algorithm': ["ball_tree", "kd_tree", "brute"],
    #'p': [1, 2]
#}
#gs = GridSearchCV(estimator=knn, cv=10, param_grid=params )
#print("accuracy:", knn.score(X_test_ml, y_test_ml))
#gsresult = gs.fit(X_train_ml, y_train_ml )
#print(gsresult.best_score_) # So you can see we increased accuracy with more than 1% for training data.
#print(gsresult.best_params_) # So this is the best possible combination for our model. Lets try with that.
#knn_best_fit = KNeighborsClassifier(algorithm = "ball_tree", leaf_size= 30, n_neighbors = 20, p=1)
#knn_best_fit.fit(X_train_ml, y_train_ml)
#y_predict_best = knn_best_fit.predict(X_test_ml)
#cm_best = confusion_matrix(y_test_ml, y_predict_best)
#cm_best
#print("accuracy:", knn.score(X_test_ml, y_predict_best))

##End of KNN... Nueral network section...
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

clf = Sequential([
    Dense(input_dim=2, units=20, activation='relu'),
    #Dropout(0.2),
    Dense(units=20, activation='relu'),
    Dense(units=2, activation='sigmoid') # final layer is output unit. Here units should be equal to the number of class in you output.
])
clf.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#clf.summary()   gives you info on whats going on
clf.fit(X_train_ml, y_train_ml, batch_size=10, epochs=50)
clf.evaluate(X_train_ml, y_train_ml)
clf.evaluate(X_test_ml, y_test_ml)

y_predict_nn = clf.predict_classes(X_test_ml)
cm_nn = confusion_matrix(y_test_ml, y_predict_nn)
cm_nn
#the nueral network performed the worst....
#the knn performed the best...
