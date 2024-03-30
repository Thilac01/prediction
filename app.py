import pandas as pd


df = pd.read_csv('data.csv')

X = df[['Age', 'BloodPressure', 'Insulin']]
y = df['Outcome']

alpha = 0.0001
iterations = 1000

c1 = 0.0
c2 = 0.0
c3 = 0.0

m = len(X)

for _ in range(iterations):
    c1_gradient = 0
    c2_gradient = 0
    c3_gradient = 0
    
    for i in range(m):
        x1 = X.iloc[i]['Age']
        x2 = X.iloc[i]['BloodPressure']
        x3 = X.iloc[i]['Insulin']

        y_pred = c1 * x1 + c2 * x2 + c3*x3

        c1_gradient += (1/m) * x1 * (y_pred - y.iloc[i])
        c2_gradient += (1/m) * x2 * (y_pred - y.iloc[i])
        c3_gradient += (1/m) * x3 * (y_pred - y.iloc[i])
    
    c1 -= alpha * c1_gradient
    c2 -= alpha * c2_gradient
    c3 -= alpha * c3_gradient

age = 43
blood_pressure = 120
Insuline = 20
prediction = c1 * age + c2 * blood_pressure + c3 * Insuline
print("Predicted diabetes percentage:", prediction)