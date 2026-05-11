# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start
2.Import required libraries: pandas, numpy, sklearn.model_selection (for train-test split), sklearn.preprocessing (for scaling, encoding).
3.Load dataset: Read Placement_Data.csv using pandas.
4.Preprocessing:
a.Drop irrelevant columns: sl_no, salary. b.Encode categorical columns into numbers (e.g., gender, stream, workex, status). c.Separate features (X) and target (y).

5.Split dataset into training and testing sets (80%-20%).
6.Standardize features using StandardScaler.
7.Initialize parameters:
a.Set learning rate α (e.g., 0.01). b.Set number of iterations (e.g., 1000). c.Initialize weights θ to zeros.

8.Define sigmoid function:
sigmoid(z) = 1 / (1 + exp(-z)).

9.Define cost function (Binary Cross-Entropy):
J(θ) = -(1/m) * Σ [ y*log(hθ(x)) + (1-y)*log(1-hθ(x)) ]

10.Gradient Descent Update Rule:
θ = θ - α * (1/m) * X.T * (hθ(x) - y)

11.Train model: Repeat gradient descent updates for given iterations.
12.Predict:
If sigmoid(Xθ) ≥ 0.5 → class = 1 (Placed). Else → class = 0 (Not Placed).

13.Evaluate performance:
a.Compute accuracy on test data. b.Display confusion matrix and classification report.

14.End

``` 

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: sanjutha D
RegisterNumber:212225240136



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
data = pd.read_csv(r"C:\Users\sanju\Downloads\Placement_Data.csv")

# Step 2: Drop unnecessary columns
if "sl_no" in data.columns:
    data = data.drop("sl_no", axis=1)

if "salary" in data.columns:
    data = data.drop("salary", axis=1)

# Step 3: Encode categorical columns
label_encoders = {}

for column in data.columns:
    if data[column].dtype == object:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Step 4: Define Features and Target
X = data.drop("status", axis=1).values
y = data["status"].values.reshape(-1, 1)

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 6: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Add Bias Term
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Step 8: Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 9: Cost Function
def compute_cost(X, y, theta):

    m = len(y)

    h = sigmoid(X.dot(theta))

    cost = -(1 / m) * np.sum(
        y * np.log(h + 1e-9) +
        (1 - y) * np.log(1 - h + 1e-9)
    )

    return cost

# Step 10: Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):

    m = len(y)

    for i in range(iterations):

        h = sigmoid(X.dot(theta))

        gradient = (1 / m) * X.T.dot(h - y)

        theta = theta - learning_rate * gradient

    return theta

# Step 11: Initialize Parameters
theta = np.zeros((X_train.shape[1], 1))

learning_rate = 0.01
iterations = 1000

# Step 12: Train Model
theta = gradient_descent(
    X_train,
    y_train,
    theta,
    learning_rate,
    iterations
)

# Step 13: Predictions
y_pred_prob = sigmoid(X_test.dot(theta))

y_pred = (y_pred_prob >= 0.5).astype(int)

# Step 14: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Step 15: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual:Not Placed", "Actual:Placed"],
    columns=["Pred:Not Placed", "Pred:Placed"]
)

print("\nConfusion Matrix:\n")
print(cm_df)

# Step 16: Classification Report
report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")
print(report)





```

## Output:


<img width="456" height="337" alt="image" src="https://github.com/user-attachments/assets/6c2d6a52-7e7b-453d-ade9-ca84d7d2fa7b" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

