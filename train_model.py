import pandas as pd
import joblib
import time

df = pd.read_csv('data.csv')

df_scaled = (df - df.mean()) / df.std()

X = df_scaled[['Age', 'BloodPressure', 'Insulin']]
y = df_scaled['Outcome']

alpha = 0.01
iterations = 1000

c1 = 0.0
c2 = 0.0
c3 = 0.0

m = len(X)

print("Training the model...")

for epoch in range(iterations):
    c1_gradient = 0
    c2_gradient = 0
    c3_gradient = 0

    for i in range(m):
        x1 = X.iloc[i]['Age']
        x2 = X.iloc[i]['BloodPressure']
        x3 = X.iloc[i]['Insulin']

        y_pred = c1 * x1 + c2 * x2 + c3 * x3

        c1_gradient += (1 / m) * x1 * (y_pred - y.iloc[i]) + 0.01 * c1
        c2_gradient += (1 / m) * x2 * (y_pred - y.iloc[i]) + 0.01 * c2
        c3_gradient += (1 / m) * x3 * (y_pred - y.iloc[i]) + 0.01 * c3

    c1 -= alpha * c1_gradient
    c2 -= alpha * c2_gradient
    c3 -= alpha * c3_gradient

    print(f"\rEpoch {epoch + 1}/{iterations} {'▋' * (epoch + 1)}", end='', flush=True)
    time.sleep(0.1)

print("\nTraining completed.")

model_params = {'c1': c1, 'c2': c2, 'c3': c3}
joblib.dump(model_params, 'linear_regression_model.pkl')
