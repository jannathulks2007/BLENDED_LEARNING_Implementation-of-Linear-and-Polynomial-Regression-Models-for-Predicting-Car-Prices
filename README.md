# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import required libraries and load the dataset using pandas.
2.  Select input features (enginesize, horsepower, citympg, highwaympg) and target variable (price).
3.  Split the dataset into training and testing sets (80% training, 20% testing).
4.  Create a Linear Regression pipeline with StandardScaler, train the model, and predict test data.
5.  Create a Polynomial Regression (degree 2) pipeline, train the model, and predict test data.
6.  Evaluate both models using MSE, MAE, and R² score.
7.  Plot actual vs predicted prices to compare Linear and Polynomial regression performance.

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())
x = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
lr=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(x_train, y_train)
y_pred_linear = lr.predict(x_test)
poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(x_train, y_train)
y_pred_poly=poly_model.predict(x_test)
print('Name:JANNATHUL SHABAN.A')
print('Reg.No:212225220043')
print("Linear Regression:")
print('MSE=',mean_squared_error(y_test,y_pred_linear))
print('MAE=',mean_absolute_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)
print('R2 Score=',r2score)
print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test,y_pred_poly):.2f}")
print(f"R2 score: {r2_score(y_test, y_pred_poly):.2f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(y_test,y_pred_poly,label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(), y.max()],'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
<img width="893" height="149" alt="image" src="https://github.com/user-attachments/assets/a921fd9b-258b-49cf-81b8-b55f284a6e04" />
<img width="982" height="98" alt="image" src="https://github.com/user-attachments/assets/47db160e-1867-4935-9aa2-7b46eab26e9f" />
<img width="1206" height="598" alt="image" src="https://github.com/user-attachments/assets/eba25917-1058-4232-acf3-df3b1ef7f759" />

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
