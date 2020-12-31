# Import Section

import numpy as np
import pandas as pd
import pickle

print('Hello Python Flask')

dataset=pd.read_csv('dataset.csv')
# print(dataset.head())

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# print(x)

# print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn import linear_model

# create object
lr = linear_model.LinearRegression()

# input the training set data to fit the curve and calculate slope and Y-intercept (m,b)
lr.fit(x_train,y_train)


# Print Model Parameters
# print m
m=lr.coef_[0]
print('Slope(m)= ',m)

# print b
b=lr.intercept_
print('Y-intercept(b)= ',b)

print('Linear Model Equation:')
print('Y = ',m,' X + ',b)


y_pred=lr.predict(x_test)
y_valid=lr.predict(x_train)

#Validation
from sklearn import metrics
print("MSE:",metrics.mean_squared_error(y_pred,y_test))
print("MAE:",metrics.mean_absolute_error(y_pred,y_test))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
print("r2_score:",metrics.r2_score(y_pred,y_test))

# Saving the model
pickle.dump(lr,open('simple.pkl','wb'))