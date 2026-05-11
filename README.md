# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program.
2.Read two input numbers from the user:
  *.First number → A
  *.Second number → B
3.Perform addition operation:
  *.Add A and B
  *.Store the sum in a variable called Result
4.Display the value of Result on the screen.
5.Stop the program.
``` 

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: sanjutha D
RegisterNumber:212225240136


import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([0, 0, 0, 1, 1])

w = 0
b = 0

learning_rate = 0.1
iterations = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(iterations):
    z = w * X + b
    Y_pred = sigmoid(z)
    
    dw = np.mean((Y_pred - Y) * X)
    db = np.mean(Y_pred - Y)
    
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Weight (w):", w)
print("Bias (b):", b)

predictions = sigmoid(w * X + b)
print("Predicted Probabilities:", predictions)
print("Predicted Classes:", [1 if i > 0.5 else 0 for i in predictions]) 
*/
```

## Output:


<img width="780" height="110" alt="image" src="https://github.com/user-attachments/assets/dfb3060f-9866-457c-a4f5-2cfb085e6793" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

