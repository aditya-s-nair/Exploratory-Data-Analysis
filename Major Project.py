                                       # # Aditya S Nair # #
import pandas as pd
import matplotlib.pyplot as plt

#Getting the data and organizing it using a dataframe.

data = pd.read_csv("https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/areavsprices.csv ")
print(data)

#Data visualization of the data in RED

plt.plot(data["Area"],data["Prices"],color="red",label="DATASET")

#To divide the data into input and output

x = data.iloc[:,0:1].values         ##Input
y = data.iloc[:,1].values           ##Output

#To Run a classifer (since no normalization is required due to a relatively clean dataset)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#To map inputs with outputs or to Fit the model

print("\n",model.fit(x,y))

#To predict the output

y_predict = model.predict(x)
print("\n",y_predict)

#Comparing prdicted output with the actual output

print("\n",y)

# Since there is a huge difference in our predicted value , we can say that our model is NON - LINEAR
## To check if the model prediction has been done properly we do the cross-verification "Using EQUATION OF A STRAIGHT LINE" i.e.  " y = mx + C " 

# y = dependent variable , x = independent variable , m = slope , C = y-intercept 

# Slope::
m = model.coef_
print("\nValue of slope is:: ",m)

# y-interccept::
C = model.intercept_    
print("\nValue of y-interccept is:: ",C)

# NOW THE EQUATION TO CROSS_VERIFY THE MODEL IS ::
#The value of 'mx+C' and y_predict (Here x = 1400)

a=m*1400 + C
print("\n<<<Actual Value>>>",a)

#        <<Prediction>>

p=model.predict([[1400]])
print("\n<<<Predicted Value>>>",p)

if a==p:
    print("\nThe actual and predicted values are the same hence the model is Accurate.")
## Plotting BEST FIT LINE

plt.scatter(x,y,color="orange")
plt.plot(x,y_predict,color="lime",label="BEST FIT LINE")
plt.legend()